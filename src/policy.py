from flax import nnx
import jax
import rlax
import distrax


class PolicyNetwork(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(in_features=obs_dim, out_features=128, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=128, out_features=128, rngs=rngs)
        self.mean = nnx.Linear(in_features=128, out_features=action_dim, rngs=rngs)
        self.std = nnx.Linear(in_features=128, out_features=action_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        mean = self.mean(x)
        std = self.std(x)

        return mean, std
    

class VisionPolicyNetwork(nnx.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)