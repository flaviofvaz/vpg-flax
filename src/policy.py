from flax import linen as nn
import jax.numpy as jnp

class MeanNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(100, name="dense1")(x)
        x = nn.tanh(x)
        x = nn.Dense(50, name="dense2")(x)
        x = nn.tanh(x)
        x = nn.Dense(25, name="dense3")(x)
        x = nn.tanh(x)
        x = nn.Dense(1, name="out")(x)
        return x

class GaussianPolicy(nn.Module):
    @nn.compact
    def __call__(self, x):
        log_std = self.param("log_std", nn.initializers.zeros, (1))
        mean = MeanNetwork()(x)
        std = jnp.exp(log_std)
    
        return mean, std

class CriticNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(100, name="dense1")(x)
        x = nn.tanh(x)
        x = nn.Dense(50, name="dense2")(x)
        x = nn.tanh(x)
        x = nn.Dense(25, name="dense3")(x)
        x = nn.tanh(x)
        x = nn.Dense(1, name="out")(x)
        return x.squeeze()
    