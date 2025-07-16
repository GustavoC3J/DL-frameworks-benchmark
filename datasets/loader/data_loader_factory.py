
import numpy as np

class DataLoaderFactory:
    def __init__(self, framework: str):
        self.framework = framework.lower()
        if self.framework not in ("torch", "tf"):
            raise ValueError(f"Unsupported framework {framework}")

    def fromNumpy(self, X, Y, batch_size, shuffle):
        if self.framework == "torch":
            import torch
            from torch.utils.data import TensorDataset, DataLoader

            if np.issubdtype(Y.dtype, np.integer):
                Y_tensor = torch.tensor(Y, dtype=torch.long)
            else:
                Y_tensor = torch.tensor(Y, dtype=torch.float32)
            
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                Y_tensor
            )

            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

        elif self.framework == "tf":
            import tensorflow as tf

            if np.issubdtype(Y.dtype, np.integer):
                y_dtype = tf.int64
            else:
                y_dtype = tf.float32

            dataset = tf.data.Dataset.from_tensor_slices(
                (tf.convert_to_tensor(X, dtype=tf.float32),
                tf.convert_to_tensor(Y, dtype=y_dtype))
            )

            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(X))

            return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

