from typing import Any
import jax
import jax.numpy as jnp
import optax
import jmp
from flax.training import train_state

from utils.precision import Precision

class TrainState(train_state.TrainState):
  batch_stats: Any
  loss_scale: jmp.LossScale


@jax.jit
def softmax_cross_entropy(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
    

@jax.jit
def accuracy(logits, y):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


@jax.jit
def mse(preds, y):
    return jnp.mean((preds - y) ** 2)


@jax.jit
def mae(preds, y):
    return jnp.mean(jnp.abs(preds - y))




@jax.jit
def classif_train_step(state: TrainState, batch, key):
    # Train step used in MLP and CNN models

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x, rngs={'dropout': key}, training=True)
        loss = softmax_cross_entropy(logits, y)

        # If it's mixed precision, scale loss
        if state.loss_scale:
            loss = state.loss_scale.scale(loss)

        return loss, logits
    
    x, y = batch

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if state.loss_scale:
        # Mixed precision: Get original loss for metrics and unscale grads for updates
        loss = state.loss_scale.unscale(loss)
        grads = state.loss_scale.unscale(grads)

        # Update scaler. If there is no NaN or Inf, apply the gradients
        grads_finite = jmp.all_finite(grads)
        new_loss_scale = state.loss_scale.adjust(grads_finite)
        state = state.replace(loss_scale=new_loss_scale)
        state = jmp.select_tree(grads_finite, state.apply_gradients(grads=grads), state)
    else:
        # No mixed precision: Update state using the gradients
        state = state.apply_gradients(grads=grads)

    metric = accuracy(logits, y)

    return state, loss, metric


@jax.jit
def classif_eval_step(state, batch):
    # Test step used in MLP and CNN models
    x, y = batch

    logits = state.apply_fn({'params': state.params}, x, training=False)

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
            training=True
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
        training=False
    )

    loss = mse(logits, y)
    metric = mae(logits, y)

    return loss, metric
