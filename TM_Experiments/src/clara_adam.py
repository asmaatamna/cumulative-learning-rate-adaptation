#---------------------------------------------------------*\
# Title: 
# Author: 
#---------------------------------------------------------*/

import jax
import jax.numpy as jnp
import optax
from typing import Any, Dict

def build_optimizer(hyperparameters: Dict[str, Any]) -> optax.GradientTransformation:
    lr = hyperparameters.get('learning_rate', 0.001)
    beta1 = hyperparameters.get('b1', 0.9)
    beta2 = hyperparameters.get('b2', 0.999)
    eps = hyperparameters.get('eps', 1e-8)
    clara_c = hyperparameters.get('clara_c', 0.2)
    clara_d = hyperparameters.get('clara_d', 1.0)

    def init_fn(params):
        return {
            'adam': optax.scale_by_adam(
                b1=beta1, b2=beta2, eps=eps).init(params),
            'path': jax.tree_map(jnp.zeros_like, params),
            'lr': lr
        }

    def update_fn(grads, state, params=None):
        # Adam step
        updates, new_adam_state = optax.scale_by_adam(
            b1=beta1, b2=beta2, eps=eps).update(grads, state['adam'], params)

        # Normalize step direction for CLARA
        step = optax.apply_updates(params, updates)
        path = jax.tree_map(
            lambda p, s: (1 - clara_c) * p + jnp.sqrt(clara_c * (2 - clara_c)) * s,
            state['path'], step
        )

        # Compute learning rate adaptation
        flat_path, _ = jax.tree_flatten(path)
        flat_updates, _ = jax.tree_flatten(updates)
        path_norm_sq = sum([jnp.sum(p**2) for p in flat_path])
        dim = sum([p.size for p in flat_path])
        lr_new = state['lr'] * jnp.exp(clara_c / (2 * clara_d) * (path_norm_sq / dim - 1))

        # Apply final scaled update
        scaled_updates = jax.tree_map(lambda u: -lr_new * u, updates)

        return scaled_updates, {
            'adam': new_adam_state,
            'path': path,
            'lr': lr_new
        }

    return optax.GradientTransformation(init_fn, update_fn)



#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\