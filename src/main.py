from environments.control import load_environment as load_control_enviroment
import numpy as np
from policy import GaussianPolicy, CriticNet
import jax
import jax.numpy as jnp
from flax.training import train_state
import numpy as np
import json
import optax
import time
from scipy.signal import lfilter
from functools import partial


# for cpu development
jax.config.update("jax_platforms", "cpu")


def discount_cumsum(arr, discount):
    return lfilter([1], [1, float(-discount)], arr[::-1], axis=0)[::-1]


class Buffer:
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.observation_buffer = np.zeros(self.get_shape(size, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros(self.get_shape(size, act_dim), dtype=np.float32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32) 
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.rewards_to_go = np.zeros(size, dtype=np.float32)
        self.state_value_buffer = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lamba = lam
        self.ptr = 0
        self.trajectory_start_idx = 0
        self.max_size = size

    def get_shape(self, size, dims):
        return (size, dims) if isinstance(dims, int) else (size, *dims) 

    def store(self, observation, action, reward, state_value):
        assert self.ptr < self.max_size

        self.observation_buffer[self.ptr] = observation
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.state_value_buffer[self.ptr] = state_value
        self.ptr += 1

    def end_of_trajectory(self, last_val=0.0):
        trajectory_slice = slice(self.trajectory_start_idx, self.ptr)
        rews = np.append(self.reward_buffer[trajectory_slice], last_val)
        vals = np.append(self.state_value_buffer[trajectory_slice], last_val)

        # GAE-Lambda Advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.advantage_buffer[trajectory_slice] = discount_cumsum(deltas, self.gamma * self.lamba)

        # rewards to go
        self.rewards_to_go[trajectory_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.trajectory_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        self.trajectory_start_idx = 0

        adv_mean = self.advantage_buffer.mean()
        adv_std = self.advantage_buffer.std()
        self.advantage_buffer = (self.advantage_buffer - adv_mean) / adv_std

        return jnp.asarray(self.observation_buffer), jnp.asarray(self.action_buffer), jnp.asarray(self.rewards_to_go), jnp.asarray(self.advantage_buffer)


def compute_actor_loss(actor_params, observations: jax.Array, actions: jax.Array, advantage: jax.Array, apply_fn):
    dist = apply_fn({"params": actor_params}, observations)
    logp = dist.log_prob(actions)

    return (-logp * advantage).mean()


def compute_v_loss(critic_params, observations, rewards_to_go, apply_fn):
    mse = (apply_fn({"params": critic_params},  observations) - rewards_to_go) ** 2
    return mse.mean()


def create_get_action(actor_apply_fn, critic_apply_fn):
    @jax.jit
    def get_action(obs: jax.Array, actor_state: train_state.TrainState, critic_state: train_state.TrainState, key: jax.random.PRNGKey):
        # actor forward apass
        dist = actor_apply_fn({"params": actor_state.params}, obs)
        
        # sample action according to distribution
        key, subkey = jax.random.split(key)
        action = dist.sample(seed=subkey)

        # critic forward pass
        v = critic_apply_fn({"params": critic_state.params}, obs)
        return action, v, key

    return get_action


def create_train_step(actor_apply_fn, critic_apply_fn):
    actor_loss_fn = partial(compute_actor_loss, apply_fn=actor_apply_fn)
    actor_grad_fn = jax.value_and_grad(actor_loss_fn, allow_int=True)
    
    critic_loss_fn = partial(compute_v_loss, apply_fn=critic_apply_fn)
    critic_grad_fn = jax.value_and_grad(critic_loss_fn)

    @jax.jit
    def train_step(actor_state, critic_state, obs, act, rew, adv):
        _, actor_grads = actor_grad_fn(actor_state.params, obs, act, adv)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        def critic_update(i, state):
            _, critic_grads = critic_grad_fn(state.params, obs, rew)
            state = state.apply_gradients(grads=critic_grads)
            return state

        critic_state = jax.lax.fori_loop(
            0, 80, critic_update, critic_state
        )
        return actor_state, critic_state 
    
    return train_step



def collect_trajectories(buffer, env, actor_state, critic_state, num_steps, max_len, random_key, get_action):
    time_step = env.reset()

    ep_len = 0
    ep_reward = 0
    ep_rewards = []
    for t in range(num_steps):
        obs = np.concatenate([time_step.observation["position"], time_step.observation["velocity"]])
        action, v, random_key = get_action(obs, actor_state, critic_state, random_key)

        next_time_step = env.step(action)
        buffer.store(obs, action, next_time_step.reward, v)

        ep_len += 1
        ep_reward += next_time_step.reward

        time_step = next_time_step

        terminal = time_step.last()
        truncate = ep_len == max_len
        last_step = t == num_steps - 1
        if terminal or truncate or last_step:

            if (truncate or last_step) and not terminal:
                # bootstrap v
                obs = np.concatenate([time_step.observation["position"], time_step.observation["velocity"]])
                action, v, random_key = get_action(obs, actor_state, critic_state, random_key)
                buffer.end_of_trajectory(v)
            else:
                buffer.end_of_trajectory(0)

            time_step = env.reset()

            ep_rewards.append(ep_reward)
            ep_reward = 0
            ep_len = 0
    
    return buffer, random_key, ep_rewards


def main():
    buffer_max_size = 50_000
    steps_per_iteration = 50_000 
    max_trajectory_size = 500
    num_iterations = 500
    env_name = "cartpole"
    task_name = "balance"
    seed = 1

    gamma = 0.99
    lam = 0.95
    learning_rate_policy = 3e-4
    learning_rate_v = 1e-3

    env = load_control_enviroment(env_name=env_name, task_name=task_name)
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    obs_dim = sum([arr.shape[0] for arr in observation_spec.values()])
    buffer = Buffer(obs_dim, action_spec.shape, buffer_max_size, gamma, lam)

    rand_key = jax.random.PRNGKey(seed=seed) 
    rand_key, actor_init_key, critic_init_key = jax.random.split(rand_key, 3)

    actor = GaussianPolicy()
    critic = CriticNet()

    actor_params = actor.init(actor_init_key, jnp.ones([1, obs_dim]))["params"]
    critic_params = critic.init(critic_init_key, jnp.ones([1, obs_dim]))["params"]

    actor_opt = optax.adam(learning_rate=learning_rate_policy)
    critic_opt = optax.adam(learning_rate=learning_rate_v)

    actor_state = train_state.TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_opt)
    critic_state = train_state.TrainState.create(apply_fn=critic.apply, params=critic_params, tx=critic_opt)

    train_step = create_train_step(
        actor_apply_fn=actor_state.apply_fn, 
        critic_apply_fn=critic_state.apply_fn
    )

    get_action = create_get_action(
        actor_apply_fn=actor_state.apply_fn, 
        critic_apply_fn=critic_state.apply_fn
    )

    training_start_time = time.time()
    for i in range(num_iterations):
        start_time = time.time()
        buffer, rand_key, ep_rewards = collect_trajectories(buffer, env, actor_state, critic_state, steps_per_iteration, max_trajectory_size, rand_key, get_action)

        avg_rewards = sum(ep_rewards) / len(ep_rewards)
        collection_time = time.time() - start_time

        start_time = time.time()
        obs, act, rew, v  = buffer.get()

        actor_state, critic_state = train_step(actor_state, critic_state, obs, act, rew, v)

        train_time = time.time() - start_time
        data = {"iteration": i, "collection_time": f"{collection_time:.2f}", "training:time": f"{train_time:.2f}", "avg_reward": f"{avg_rewards:.2f}"}
        print(json.dumps(data))
    print(f"Training done. Total training time: {(time.time() - training_start_time) // 60} minutes")


if __name__ == "__main__":
    main()
