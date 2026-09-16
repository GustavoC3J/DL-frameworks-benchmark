
from enum import Enum


class Precision(str, Enum):
    FP32 = "fp32",
    FP16 = "fp16",
    BF16 = "bf16",
    MIXED_FP16 = "mixed_fp16",
    MIXED_BF16 = "mixed_bf16"

    # By default "Precision.FP32" is returned, but only the string is needed
    def __str__(self):
        return self.value


def get_keras_precision(precision: Precision) -> str:
    res = ""

    if precision == Precision.FP32:
        res = "float32"
    elif precision == Precision.FP16:
        res = "float16"
    elif precision == Precision.BF16:
        res = "bfloat16"
    elif precision == Precision.MIXED_FP16:
        res = "mixed_float16"
    elif precision == Precision.MIXED_BF16:
        res = "mixed_bfloat16"

    return res


# torch and jmp are imported inside the functions: each environment only has its own framework
def get_torch_precision(precision: Precision):
    """(dtype, amp_dtype): parameters and computation in dtype, or float32 parameters with autocast to amp_dtype."""
    import torch

    dtypes = {
        Precision.FP32: (torch.float32, None),
        Precision.FP16: (torch.float16, None),
        Precision.BF16: (torch.bfloat16, None),
        Precision.MIXED_FP16: (torch.float32, torch.float16),
        Precision.MIXED_BF16: (torch.float32, torch.bfloat16),
    }

    if precision not in dtypes:
        raise ValueError("Unsupported precision: " + precision)

    return dtypes[precision]


def get_jmp_policy(precision: Precision):
    """(policy, loss_scale): the loss scale is only needed in mixed precision."""
    import jax.numpy as jnp
    import jmp

    if precision in (Precision.FP32, Precision.FP16, Precision.BF16):
        names = {
            Precision.FP32: "float32",
            Precision.FP16: "float16",
            Precision.BF16: "bfloat16"
        }
        return jmp.get_policy(names[precision]), None

    elif precision in (Precision.MIXED_FP16, Precision.MIXED_BF16):
        compute_dtype = jnp.float16 if precision == Precision.MIXED_FP16 else jnp.bfloat16
        policy = jmp.Policy(compute_dtype=compute_dtype, param_dtype=jnp.float32, output_dtype=jnp.float32)
    
        return policy, jmp.DynamicLossScale(jnp.float32(2 ** 15))
        
    else:
        raise ValueError("Unsupported precision: " + precision)

    