
import copy
import time

import torch

from runners.model_builder.torch_model_builder import TorchModelBuilder
from runners.runner import Runner
from utils.precision import get_torch_precision
from utils.torch_utils import adjust_outputs


class TorchRunner(Runner):

    data_framework = "torch"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # It is always 0, independently of CUDA index
        self.device = torch.device("cuda:0")

        # Fix the seed: weight initialization and dropout
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Set global floating point precision
        self.dtype, self.amp_dtype = get_torch_precision(self.precision)
        self.amp = self.amp_dtype is not None # AMP = Automatic Mixed Precision

        if self.amp:
            self.scaler = torch.amp.GradScaler()


    def define_model(self):
        self.model, self.config = TorchModelBuilder(self.model_type, self.model_complexity).build()

        # Move the model to the GPU and set its precision
        self.model.to(device=self.device, dtype=self.dtype)


    def __train_step(self, batch_x, batch_y):
        """One training step: forward, backward and update.
        
        Shared by the training loop and the
        warm-up, so both go through exactly the same path, AMP included."""

        # Send data to GPU and set dtype
        # If output has to be an integer, then batch_y dtype is not modified
        batch_x = batch_x.to(device=self.device, dtype=self.dtype)
        batch_y = batch_y.to(device=self.device, dtype=torch.int64 if batch_y.dtype == torch.int64 else self.dtype)

        self.config["optimizer"].zero_grad()

        if self.amp:
            # Get outputs and loss using lower precision
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                outputs = self.model(batch_x)

                if self.model_type == "lstm":
                    outputs = adjust_outputs(outputs, batch_y)

                loss = self.config["loss_fn"](outputs, batch_y)
                metric = self.config["metric_fn"](outputs, batch_y)

            # Perform updates in higher precision
            self.scaler.scale(loss).backward()
            self.scaler.step(self.config["optimizer"])
            self.scaler.update()

        else:
            # Get loss and perform updates using the same precision
            outputs = self.model(batch_x)

            if self.model_type == "lstm":
                outputs = adjust_outputs(outputs, batch_y)

            loss = self.config["loss_fn"](outputs, batch_y)
            metric = self.config["metric_fn"](outputs, batch_y)

            loss.backward()
            self.config["optimizer"].step()

        # Kept on the GPU and read once per epoch to avoid a synchronization per batch
        return loss.detach().float(), metric.detach().float()


    def _train(self, train_dl, val_dl):

        metric_name = self.config["metric_name"]
        best_model_weights = None
        best_val_loss = float('inf')
        self.best_epoch = 0

        history = {
            "loss": [],
            metric_name: [],
            "val_loss": [],
            f"val_{metric_name}": [],
            "epoch_time": []
        }

        # Training start time
        start_time = time.time()

        # Entrenamiento
        for epoch in range(self.epochs):
            # Training
            epoch_start_time = time.time()
            train_losses = []
            train_metrics = []

            # Set training mode
            self.model.train()
            
            for batch_x, batch_y in train_dl:
                loss, metric = self.__train_step(batch_x, batch_y)

                train_losses.append(loss)
                train_metrics.append(metric)

            # Validation
            val_loss, val_metric, _ = self.__evaluate(val_dl, True)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch


            # Save metrics
            history["loss"].append(torch.stack(train_losses).mean().item())
            history[metric_name].append(torch.stack(train_metrics).mean().item())
            history["val_loss"].append(val_loss)
            history[f"val_{metric_name}"].append(val_metric)
            history["epoch_time"].append(time.time() - epoch_start_time)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {history['loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val {metric_name}: {val_metric:.4f}")

        # Load the best model
        if (best_model_weights != None):
            self.model.load_state_dict(best_model_weights)

        return history


    def _precompile(self, train_batches, val_batches):

        # Kept to restore it once the kernels are warmed up. Dropout draws from the RNG states
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.config["optimizer"].state_dict())
        scaler_state = copy.deepcopy(self.scaler.state_dict()) if self.amp else None
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        self.model.train()
        for batch_x, batch_y in train_batches:
            self.__train_step(batch_x, batch_y)

        # Evaluation path: forward only, under no_grad
        self.__evaluate(val_batches, True)

        # Wait for the GPU before the caller stops its timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Undo everything the steps above changed
        self.model.load_state_dict(model_state)
        self.config["optimizer"].load_state_dict(optimizer_state)

        if self.amp:
            self.scaler.load_state_dict(scaler_state)

        torch.set_rng_state(rng_state)

        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


    def save(self, path):
        torch.save(self.model.state_dict(), path + f'/{self.best_epoch:02d}_model.pt')


    def __evaluate(self, test_dl, val = False):

        losses = []
        metrics = []

        # Set evaluation mode
        self.model.eval()

        start_time = time.time()

        with torch.no_grad():
            for batch_x, batch_y in test_dl:
                # Send data to GPU
                batch_x = batch_x.to(device=self.device, dtype=self.dtype)
                batch_y = batch_y.to(device=self.device, dtype=torch.long if batch_y.dtype == torch.long else self.dtype)

                if self.amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        test_outputs = self.model(batch_x)

                        if self.model_type == "lstm":
                            test_outputs = adjust_outputs(test_outputs, batch_y)

                        loss = self.config["loss_fn"](test_outputs, batch_y)
                        metric = self.config["metric_fn"](test_outputs, batch_y)

                else:
                    test_outputs = self.model(batch_x)

                    if self.model_type == "lstm":
                        test_outputs = adjust_outputs(test_outputs, batch_y)

                    loss = self.config["loss_fn"](test_outputs, batch_y)
                    metric = self.config["metric_fn"](test_outputs, batch_y)

                losses.append(loss.float())
                metrics.append(metric.float())

        # Calculate mean
        test_loss = torch.stack(losses).mean().item()
        test_metric = torch.stack(metrics).mean().item()
    
        # Print log message if it is test
        if not val:
            print(f"Loss: {test_loss:.4f} - {self.config['metric_name']}: {test_metric:.4f}")

        return (
            test_loss,
            test_metric,
            time.time() - start_time
        )



    def evaluate(self, testX, testY):
        test_dl = self.dl_factory.fromNumpy(testX, testY, self.batch_size, shuffle=False)

        return self.__evaluate(test_dl)



