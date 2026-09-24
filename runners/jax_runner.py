
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint
from flax.training import checkpoints

from runners.model_builder.flax_model_builder import FlaxModelBuilder
from runners.runner import Runner
from utils.jax_utils import TrainState, make_eval_step, make_train_step
from utils.precision import get_jmp_policy


class JaxRunner(Runner):

    data_framework = "torch"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Fix the seed
        self.key = jax.random.key(seed=self.seed)

        # Set GPUs
        if len(self.gpu_ids) == 1:
            jax.config.update("jax_default_device", jax.devices("gpu")[0])

        # Set global floating point precision
        self.policy, self.loss_scale = get_jmp_policy(self.precision)


    def define_model(self):
        self.key, subkey = jax.random.split(self.key)
        self.model, self.config = FlaxModelBuilder(self.model_type, self.model_complexity, subkey, self.policy).build()

        # Set up the state using the model and cofiguration
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.config["params"],
            tx=self.config["optimizer"],
            batch_stats=self.config.get("batch_stats", None),
            loss_scale=self.loss_scale
        )

        # Built once: a new wrapper would be traced and compiled again on every epoch
        self.train_step = make_train_step(self.config["loss_fn"], self.config["metric_fn"])
        self.eval_step = make_eval_step(self.config["loss_fn"], self.config["metric_fn"])


    def _train(self, train_dl, val_dl):
        
        metric_name = self.config["metric_name"]
        best_model_weights = None
        best_val_loss = float('inf')

        history = {
            "loss": [],
            metric_name: [],
            "val_loss": [],
            f"val_{metric_name}": [],
            "epoch_time": []
        }

        # Training start time
        start_time = time.time()

        for epoch in range(self.epochs):
            # Training
            epoch_start_time = time.time()
            train_losses = []
            train_metrics = []
            
            for batch_x, batch_y in train_dl:
                # Parse data into JAX arrays
                batch_x = jnp.array(batch_x)
                batch_y = jnp.array(batch_y)

                self.key, subkey = jax.random.split(self.key)
                self.state, loss, metric = self.train_step(self.state, (batch_x, batch_y), subkey)
                train_losses.append(loss)
                train_metrics.append(metric)


            # Validation
            val_loss, val_metric, _ = self.__evaluate(val_dl, True)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = self.state.replace()

            # Save metrics
            history["loss"].append(jnp.mean(jnp.array(train_losses)).item())
            history[metric_name].append(jnp.mean(jnp.array(train_metrics)).item())
            history["val_loss"].append(val_loss)
            history[f"val_{metric_name}"].append(val_metric)
            history["epoch_time"].append(time.time() - epoch_start_time)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {history['loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val {metric_name}: {val_metric:.4f}")

        # Load the best model
        if (best_model_weights != None):
            self.state = best_model_weights
        
        return history


    def _precompile(self, train_batches, val_batches):

        # The state is immutable and every step returns a new one, so keeping the reference
        # is all it takes to undo the warm-up
        state, key = self.state, self.key

        for batch_x, batch_y in train_batches:
            self.key, subkey = jax.random.split(self.key)
            self.state, _, _ = self.train_step(self.state, (jnp.array(batch_x), jnp.array(batch_y)), subkey)

        jax.block_until_ready(self.state)

        # Compiles eval_step, which validation and the test reuse
        self.__evaluate(val_batches, True)

        self.state, self.key = state, key


    def save(self, path):
        checkpoints.save_checkpoint(
            Path(path).absolute(),
            self.state.replace(loss_scale=None), # Not needed anymore, and it's incompatible with checkpointing
            0,
            orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
        )


    def __evaluate(self, test_dl, val = False):

        test_loss = 0
        test_metric = 0
        num_batches = len(test_dl)
        
        start_time = time.time()
        for batch_x, batch_y in test_dl:
            loss, metric = self.eval_step(self.state, (jnp.array(batch_x), jnp.array(batch_y)))

            test_loss += loss
            test_metric += metric

        # Calculate mean
        test_loss /= num_batches
        test_metric /= num_batches
        
        # Print log message if it is test
        if not val:
            print(f"Loss: {test_loss.item():.4f} - {self.config['metric_name']}: {test_metric.item():.4f}")

        return (
            test_loss.item(),
            test_metric.item(),
            time.time() - start_time
        )

    
    def evaluate(self, testX, testY):
        test_dl = self.dl_factory.fromNumpy(testX, testY, self.batch_size, shuffle=False)

        return self.__evaluate(test_dl)
        



