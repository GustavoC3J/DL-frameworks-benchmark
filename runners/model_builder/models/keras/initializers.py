
import keras
from keras import initializers, ops


# Since Keras 3.9, load_model only imports keras modules, so custom objects must be registered
@keras.saving.register_keras_serializable()
class Float32Orthogonal(initializers.Orthogonal):
    """Orthogonal initializer whose QR decomposition runs in float32.

    QR has no float16 or bfloat16 kernel, so the plain initializer fails in those precisions.
    """

    def __call__(self, shape, dtype=None):
        return ops.cast(super().__call__(shape, "float32"), dtype or keras.config.floatx())
