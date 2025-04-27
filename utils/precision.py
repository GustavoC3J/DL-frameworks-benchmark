
from enum import Enum


class Precision(str, Enum):
    FP32 = "fp32",
    FP16 = "fp16",
    BF16 = "bf16",
    MIXED_FP16 = "mixed_fp16",
    MIXED_BF16 = "mixed_bf16"


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
