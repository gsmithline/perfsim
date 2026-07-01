import jax
import jax.numpy as jnp


def linear_schedule(count, total_counts, init_val, final_val):
    frac = jnp.clip(count / total_counts, min=0.0, max=1.0)
    return init_val + (final_val - init_val) * frac

def exponential_schedule(count, total_counts, init_val, final_val):
    frac = jnp.clip(count / total_counts, min=0.0, max=1.0)
    return init_val * (final_val / init_val) ** frac

def exponential_linear_schedule(count, total_counts, init_val, final_val, epsilon, switch_frac):
    exponential_total_counts = switch_frac * total_counts
    linear_total_counts = total_counts - exponential_total_counts
    frac = jnp.clip(count / total_counts, min=0.0, max=1.0)
    return jax.lax.select(
        frac < switch_frac,
        exponential_schedule(count, exponential_total_counts, init_val, epsilon),
        linear_schedule(count - exponential_total_counts, linear_total_counts, epsilon, final_val)
    )

def sigmoid_schedule(count, total_counts, init_val, final_val, a, b, c):
    frac = jnp.clip(count / total_counts, min=0.0, max=1.0)
    ramp = a / (1 + jnp.exp(-b * (frac - c)))
    return jnp.clip(
        init_val + (final_val - init_val) * ramp,
        min = init_val,
        max = final_val,
    )
