import flax.nnx
from environments.control import load_environment as load_control_enviroment
import numpy as np
from policy import GaussianPolicy, Baseline
import jax
import jax.numpy as jnp
import flax
import numpy as np
import json
from flax import nnx
import optax
import time


# for cpu development
jax.config.update("jax_platforms", "cpu")


class Buffer:
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.observation_buffer = jnp.zeros(self.get_shape(size, obs_dim), dtype=jnp.float32)
        self.action_buffer = jnp.zeros(self.get_shape(size, act_dim), dtype=jnp.float32)
        self.reward_buffer = jnp.zeros(size, dtype=jnp.float32)
        self.rewards_to_go = jnp.zeros(size, dtype=jnp.float32)
        self.state_value_buffer = jnp.zeros(size, dtype=jnp.float32)

        self.gamma = gamma
        # self.lam = lam
        self.ptr = 0
        self.trajectory_start_idx = 0
        self.max_size = size

    def get_shape(self, size, dims):
        return (size, dims) if isinstance(dims, int) else (size, *dims) 

    def store(self, observation, action, reward, state_value):
        assert self.ptr < self.max_size

        self.observation_buffer = self.observation_buffer.at[self.ptr].set(observation)
        self.action_buffer = self.action_buffer.at[self.ptr].set(action)
        self.reward_buffer = self.reward_buffer.at[self.ptr].set(reward)
        self.state_value_buffer = self.state_value_buffer.at[self.ptr].set(state_value)
        self.ptr += 1

    def end_of_trajectory(self, last_val=0.0):
        trajectory_slice = slice(self.trajectory_start_idx, self.ptr)

        rewards = jnp.append(self.reward_buffer[trajectory_slice], last_val)
        rewards_reversed = rewards[::-1]

        discount_factors = self.gamma ** jnp.arange(len(rewards))
        discounted_rewards = rewards_reversed / discount_factors
        cumsum = jnp.cumsum(discounted_rewards)

        rew_to_go = cumsum * discount_factors

        # reverse
        rew_to_go = rew_to_go[::-1]

        self.rewards_to_go = self.rewards_to_go.at[trajectory_slice].set(rew_to_go[:-1])
        self.trajectory_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        self.trajectory_start_idx = 0

        return self.observation_buffer, self.action_buffer, self.rewards_to_go, self.state_value_buffer


def compute_loss(policy, observations, actions, rewards, v):
    dist = policy(jnp.asarray(observations))
    logp = dist.log_prob(jnp.asarray(actions))

    weights = rewards - v
    return (-logp * weights).mean()


def compute_v_loss(baseline, observations, rewards):
    mse = (baseline(observations) - rewards) ** 2
    return mse.mean()


@nnx.jit
def get_action(obs, policy, baseline, key):
    dist = policy(obs)
    key, subkey = jax.random.split(key)
    action = dist.sample(seed=subkey)

    v = baseline(obs)
    return action, v, key


def collect_trajectories(buffer: Buffer, env, policy, baseline, num_steps, max_len, random_key):
    time_step = env.reset()

    ep_len = 0
    ep_reward = 0
    ep_rewards = []
    for t in range(num_steps):
        obs = np.concatenate([time_step.observation["position"], time_step.observation["velocity"]])
        action, v, random_key = get_action(obs, policy, baseline, random_key)

        next_time_step = env.step(action)
        buffer.store(obs, action, next_time_step.reward, v.item())

        ep_len += 1
        ep_reward += next_time_step.reward

        time_step = next_time_step

        terminal = time_step.last()
        truncate = ep_len == max_len
        last_step = t == num_steps - 1
        if terminal or truncate or last_step:
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

    random_key = jax.random.PRNGKey(seed=seed) 
    random_key, random_subkey_policy, random_subkey_baseline = jax.random.split(random_key, 3)
    policy = GaussianPolicy(obs_dim=obs_dim, action_dim=action_spec.shape[0], rngs=flax.nnx.Rngs(params=random_subkey_policy))
    baseline = Baseline(obs_dim=obs_dim, out_dim=1, rngs=flax.nnx.Rngs(params=random_subkey_baseline))

    tx_actor = optax.sgd(learning_rate=learning_rate_policy)
    tx_critic = optax.sgd(learning_rate=learning_rate_v)
    optimizer_actor = nnx.Optimizer(model=policy, tx=tx_actor)
    optimizer_critic = nnx.Optimizer(model=baseline, tx=tx_critic)

    training_start_time = time.time()
    for i in range(num_iterations):
        start_time = time.time()
        buffer, random_key, ep_rewards = collect_trajectories(buffer, env, policy, baseline, steps_per_iteration, max_trajectory_size, random_key)

        avg_rewards = sum(ep_rewards) / len(ep_rewards)
        collection_time = time.time() - start_time

        start_time = time.time()
        obs, act, rew, v  = buffer.get()

        graphdef_policy, state_policy = nnx.split((policy, optimizer_actor))
        graphdef_critic, state_critic = nnx.split((baseline, optimizer_critic))

        state_policy, state_critic = train_step(obs, act, rew, v, graphdef_policy, state_policy, graphdef_critic, state_critic)

        nnx.update((policy, optimizer_actor), state_policy)
        nnx.update((baseline, optimizer_critic), state_critic)
        train_time = time.time() - start_time

        data = {"iteration": i, "collection_time": f"{collection_time:.2f}", "training:time": f"{train_time:.2f}", "avg_reward": f"{avg_rewards:.2f}"}
        print(json.dumps(data))
    print(f"Training done. Total training time: {(time.time() - training_start_time) // 60} minutes")


@jax.jit
def train_step(obs, act, rew, v, graphdef_policy, state_policy, graphdef_critic, state_critic):
    policy, optimizer_actor = nnx.merge(graphdef_policy, state_policy)

    grad_fn_policy = nnx.value_and_grad(compute_loss)
    grad_fn_v = nnx.value_and_grad(compute_v_loss)
    _, grads = grad_fn_policy(policy, obs, act, rew, v)
    optimizer_actor.update(grads)

    state_policy = nnx.state((policy, optimizer_actor))

    def critic_update(i, carry):
        graphdef_critic, state_critic = carry
        critic, critic_opt = nnx.merge(graphdef_critic, state_critic)
        _, critic_grads = grad_fn_v(critic, obs, rew)
        critic_opt.update(critic_grads)
        state_critic = nnx.state((critic, critic_opt))
        return graphdef_critic, state_critic
    
    init_carry = (graphdef_critic, state_critic)
    graphdef_critic, state_critic = jax.lax.fori_loop(
        0, 80, critic_update, init_carry
    )
    return state_policy, state_critic


if __name__ == "__main__":
    main()
