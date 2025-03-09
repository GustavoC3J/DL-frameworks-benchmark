from typing import Any
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

class TrainState(train_state.TrainState):
  batch_stats: Any

def softmax_cross_entropy(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
    

def accuracy(logits, y):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


def mse(preds, y):
    return jnp.mean((preds - y) ** 2)


def mae(preds, y):
    return jnp.mean(jnp.abs(preds - y))




@jax.jit
def classif_train_step(state, batch, key):
    # Train step used in MLP and CNN models
    x, y = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x, rngs={'dropout': key}, deterministic=False)
        return softmax_cross_entropy(logits, y), logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    metric = accuracy(logits, y)

    return state, loss, metric


@jax.jit
def classif_eval_step(state, batch):
    # Test step used in MLP and CNN models
    x, y = batch

    logits = state.apply_fn({'params': state.params}, x, deterministic=True)

    loss = softmax_cross_entropy(logits, y)
    metric = accuracy(logits, y)

    return loss, metric


@jax.jit
def regression_train_step(state, batch, key):
    # Train step used in MLP and CNN models
    x, y = batch

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x,
            rngs={'dropout': key},
            mutable=['batch_stats'],
            deterministic=False
        )
        
        loss = mse(logits, y)
        aux = (logits, updates)
        return loss, aux

    (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    metric = mae(logits, y)

    return state, loss, metric


@jax.jit
def regression_eval_step(state, batch):
    # Test step used in MLP and CNN models
    x, y = batch

    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        x,
        deterministic=True
    )

    loss = mse(logits, y)
    metric = mae(logits, y)

    return loss, metric
