import numpy as np


def blend(lower: np.ndarray, upper: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """Blend two images by a fixed linear ratio."""
    return (ratio * lower.astype(float) + (1 - ratio) * upper.astype(float)).astype(int)


def blend_rgba(*imgs: np.ndarray) -> np.ndarray:
    """Alpha-composite one or more RGBA images from front to back."""
    if len(imgs) == 0:
        raise ValueError("At least one image is required for blending")
    elif len(imgs) == 1:
        return imgs[0]
    elif len(imgs) == 2:
        fg, bg = imgs
    else:
        fg = imgs[0]
        bg = blend_rgba(*imgs[1:])

    fg_rgb, fg_alpha = fg[..., :3], fg[..., 3:4]
    bg_rgb, bg_alpha = bg[..., :3], bg[..., 3:4]
    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_rgb = (fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)) / np.maximum(
        out_alpha, 1e-8
    )
    blended = np.concatenate([out_rgb, out_alpha], axis=-1)
    return blended


def color_burn(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Apply the color burn blend mode to two 8-bit images."""
    A = lower.astype(float) / 255
    B = upper.astype(float) / 255
    result = np.zeros_like(A)
    mask = B > 0
    result[mask] = 1 - (1 - A[mask]) / B[mask]
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Apply the multiply blend mode to two 8-bit images."""
    A = first.astype(float) / 255
    B = second.astype(float) / 255
    return ((A * B) * 255).astype(int)
