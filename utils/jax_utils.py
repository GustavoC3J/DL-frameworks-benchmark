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


def softmax_cross_entropy(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
    

def accuracy(logits, y):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


def mse(preds, y):
    return jnp.mean((preds - y) ** 2)


def mae(preds, y):
    return jnp.mean(jnp.abs(preds - y))




# Can't pass functions as parameter to a jit function, so this wrapper sets it up
def make_train_step(loss_fn, metric_fn):

    @jax.jit
    def train_step(state: TrainState, batch, key):
        
        x, y = batch

        def loss_wrapper(params):
            variables = {'params': params}
            mutable = False

            if state.batch_stats is not None:
                variables['batch_stats'] = state.batch_stats
                mutable = ['batch_stats']

            outputs = state.apply_fn(
                variables,
                x,
                rngs={'dropout': key},
                mutable=mutable,
                training=True
            )

            if state.batch_stats is not None:
                logits, updates = outputs
            else:
                logits = outputs
                updates = None

            loss = loss_fn(logits, y)
            # If it's mixed precision, scale loss
            if state.loss_scale:
                loss = state.loss_scale.scale(loss)

            aux = (logits, updates)
            return loss, aux
        

        (loss, (logits, updates)), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(state.params)

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

        if state.batch_stats is not None:
            state = state.replace(batch_stats=updates['batch_stats'])

        metric = metric_fn(logits, y)

        return state, loss, metric
    
    return train_step


def make_eval_step(loss_fn, metric_fn):

    @jax.jit
    def eval_step(state, batch):
        # Test step used in MLP and CNN models
        x, y = batch

        variables = {'params': state.params}

        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats

        logits = state.apply_fn(
            variables,
            x,
            training=False
        )

        loss = loss_fn(logits, y)
        metric = metric_fn(logits, y)

        return loss, metric
    
    return eval_step
