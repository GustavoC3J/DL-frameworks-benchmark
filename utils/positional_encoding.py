import numpy as np


def sinusoidal_encoding(length, d_model):
    """The fixed positional encoding of "Attention Is All You Need": sines in the even dimensions
    and cosines in the odd ones. Computed once in numpy so the three frameworks add the same values."""
    positions = np.arange(length)[:, None]
    frequencies = np.exp(-np.log(10000.0) * np.arange(0, d_model, 2) / d_model)

    encoding = np.zeros((length, d_model))
    encoding[:, 0::2] = np.sin(positions * frequencies)
    encoding[:, 1::2] = np.cos(positions * frequencies)

    return encoding.astype("float32")
