from flax import nnx
import jax
import jax.numpy as jnp
from distrax import MultivariateNormalDiag


class MeanNetwork(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(in_features=obs_dim, out_features=100, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=100, out_features=50, rngs=rngs)
        self.linear3 = nnx.Linear(in_features=50, out_features=25, rngs=rngs)

        self.mean = nnx.Linear(in_features=25, out_features=action_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.tanh(self.linear1(x))
        x = nnx.tanh(self.linear2(x))
        x = self.linear3(x)
        mean = self.mean(x)

        return mean
    

class Baseline(nnx.Module):
    def __init__(self, obs_dim: int, out_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(in_features=obs_dim, out_features=out_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.linear(x)
        return x
    

class GaussianPolicy(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.network = MeanNetwork(obs_dim, action_dim, rngs)
        self.log_std = nnx.Param(jnp.zeros((action_dim)))

    def __call__(self, x: jax.Array):
        mean = self.network(x)
        std = jnp.exp(self.log_std.value)
        dist = MultivariateNormalDiag(loc=mean, scale_diag=std)

        return dist
    



class VisionPolicyNetwork(nnx.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)