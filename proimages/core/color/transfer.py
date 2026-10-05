import numpy as np

# IEC 61966-2-1: a linear segment up to each threshold, a power segment above it
_DECODE_THRESHOLD = 0.04045
_ENCODE_THRESHOLD = 0.0031308


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Decode display-encoded sRGB to linear light, element-wise, in float32.

    Values below 0 follow the linear segment and values above 1 the power segment; nothing is clipped (callers
    clip). The result is never NaN for finite input and raises no RuntimeWarning, unless the result itself no
    longer fits float32 (an input above about 1e16).
    """
    x = np.asarray(x, dtype=np.float32)
    # np.where evaluates both branches everywhere, so the power is fed a clamped base: a negative one would be NaN
    power = ((np.maximum(x, _DECODE_THRESHOLD) + 0.055) / 1.055) ** 2.4
    return np.where(x <= _DECODE_THRESHOLD, x / 12.92, power).astype(np.float32, copy=False)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Encode linear light to display-encoded sRGB, element-wise, in float32.

    Values below 0 follow the linear segment and values above 1 the power segment; nothing is clipped (callers
    clip). The result is never NaN for finite input and raises no RuntimeWarning, unless the result itself no
    longer fits float32 (an input below about -2.6e37).
    """
    x = np.asarray(x, dtype=np.float32)
    # np.where evaluates both branches everywhere, so each is fed a clamped input: a negative base to the power
    # would be NaN, and 12.92 * x would overflow for a huge x although the power branch is the one selected
    linear = 12.92 * np.minimum(x, _ENCODE_THRESHOLD)
    power = 1.055 * np.maximum(x, _ENCODE_THRESHOLD) ** (1 / 2.4) - 0.055
    return np.where(x <= _ENCODE_THRESHOLD, linear, power).astype(np.float32, copy=False)
