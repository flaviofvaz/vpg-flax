import flax.nnx
from environments.control import load_environment as load_control_enviroment
import numpy as np
from policy import PolicyNetwork
import jax
import jax.numpy as jnp
import flax
import distrax
import reverb
import tensorflow as tf
import numpy as np


# for cpu development
jax.config.update("jax_platforms", "cpu")

def init_client(buffer_name: str, max_size: int, min_size: int):
    server = reverb.Server(
        tables=[
            reverb.Table(
                name=buffer_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=max_size,
                rate_limiter=reverb.rate_limiters.MinSize(min_size),
                signature={
                    "obsevartions": tf.TensorSpec([5], tf.float32),
                    "actions": tf.TensorSpec([], tf.float32),
                    #"reward": tf.TensorSpec([], tf.float32),
                }
            ),
        ]
    )
    client = reverb.Client(f"localhost:{server.port}")
    return server, client

def main():
    buffer_name = "trajectories"
    buffer_min_size = 1
    buffer_max_size = 1000
    trajectory_size = 100
    trajectory_num = 128 
    env_name = "cartpole"
    task_name = "swingup"
    seed = 32

    batch_size= 32
    learn_steps = 100

    server, client = init_client(buffer_name, buffer_max_size, buffer_min_size)
    env = load_control_enviroment(env_name=env_name, task_name=task_name)

    action_spec = env.action_spec()

    random_key = jax.random.PRNGKey(seed=seed) 
    random_key, random_subkey = jax.random.split(random_key)
    policy_net = PolicyNetwork(obs_dim=5, action_dim=action_spec.shape[0], rngs=flax.nnx.Rngs(params=random_subkey))

    time_step = env.reset()
    with client.trajectory_writer(num_keep_alive_refs=trajectory_size) as writer:
        for i in range(trajectory_num):
            count = 0
            while count < trajectory_size:
                obs = np.concat([time_step.observation["position"], time_step.observation["velocity"]])
                mean, std = policy_net(obs)
                dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
                random_key, random_subkey = jax.random.split(random_key)
                action = dist.sample(seed=random_subkey)

                writer.append({"action": action, "observation": obs})
                time_step = env.step(action)
                count += 1

                if time_step.last():
                    time_step = env.reset()

            writer.create_item(
                table=buffer_name,
                priority=1.0,
                trajectory={
                    "actions": writer.history["action"][-count:],
                    "observations": writer.history["observation"][-count:],
                }
            )

    
    dataset = reverb.TrajectoryDataset.from_table_signature(
                                            server_address=client.server_address,
                                                table=buffer_name,
        max_in_flight_samples_per_worker=10,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for i in range(learn_steps):



if __name__ == "__main__":
    main()
