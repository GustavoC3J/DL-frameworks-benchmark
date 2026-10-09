import keras
from keras import ops


# Since Keras 3.9, load_model only imports keras modules, so custom objects must be registered
@keras.saving.register_keras_serializable()
class Adam(keras.optimizers.Adam):
    """Adam whose bias correction runs in float32, as in torch and optax.

    Keras computes it in the variable's dtype, and bfloat16 rounds beta_2 = 0.999 to 1: the step size
    is zero and bfloat16 models never train. The moments are still kept in the variable's dtype.
    """

    def update_step(self, gradient, variable, learning_rate):
        local_step = ops.cast(self.iterations + 1, "float32")
        beta_1_power = ops.power(ops.cast(self.beta_1, "float32"), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, "float32"), local_step)

        alpha = ops.cast(learning_rate, "float32") * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        alpha = ops.cast(alpha, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        # From here on, the same as keras.optimizers.Adam.update_step
        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), 1 - self.beta_1)
        )
        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - self.beta_2
            ),
        )
        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(v_hat, ops.maximum(v_hat, v))
            v = v_hat
        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)
            ),
        )
