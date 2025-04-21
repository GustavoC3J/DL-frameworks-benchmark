
from enum import Enum


class Precision(str, Enum):
    FP32 = "fp32",
    MIXED_PRECISION = "mixed",
    BF16 = "bf16"


def get_keras_precision(precision: Precision) -> str:
    res = ""

    if precision == Precision.FP32:
        res = "float32"
    elif precision == Precision.MIXED_PRECISION:
        res = "mixed_bfloat16"
    elif precision == Precision.BF16:
        res = "bfloat16"

    return res
