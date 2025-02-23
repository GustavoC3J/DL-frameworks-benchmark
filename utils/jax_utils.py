import jax
import jax.numpy as jnp
import optax



def softmax_cross_entropy(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
    

def accuracy(logits, y):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)



@jax.jit
def mlp_train_step(state, batch, key):
    x, y = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x, rngs={'dropout': key}, deterministic=False)
        return softmax_cross_entropy(logits, y), logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    metric = accuracy(logits, y)

    return state, loss, metric


@jax.jit
def mlp_eval_step(state, batch):
    x, y = batch

    logits = state.apply_fn({'params': state.params}, x, deterministic=True)

    loss = softmax_cross_entropy(logits, y)
    metric = accuracy(logits, y)

    return loss, metric