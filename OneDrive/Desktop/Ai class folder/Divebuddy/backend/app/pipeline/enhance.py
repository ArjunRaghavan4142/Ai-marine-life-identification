"""Underwater image enhancement for dive footage.

Water absorbs red light first, so dive video skews blue-green and loses
contrast to backscatter/haze. That colour cast and low contrast is a big reason
detections come back at 0.5-0.7 confidence and species IDs land on "Unknown" --
the models never see the fish's real colours. This restores each frame before it
reaches the detectors and the identifier:

  1. Grey-world white balance -- rescales each channel toward a common grey mean.
     Underwater the red mean is the lowest, so red gets boosted the most: the
     exact colour the water stole is put back.
  2. CLAHE contrast on the L (lightness) channel in LAB -- cuts haze and lifts
     local detail without shifting hue or blowing out highlights.

Deliberately cheap (a few ms/frame, pure OpenCV/NumPy) so it barely moves total
runtime. `strength` blends the result back toward the original, so clear-water
clips can dial the correction down instead of over-saturating.
"""
import cv2
import numpy as np

# Master switch + default blend. Applied to every sampled frame before detection,
# so both the detectors and the crops handed to the identifier see corrected
# colour. Set ENHANCE_FRAMES=False to detect on raw frames (A/B comparison).
ENHANCE_FRAMES = True
DEFAULT_STRENGTH = 0.85


def _white_balance(bgr: np.ndarray) -> np.ndarray:
    """Grey-world white balance. Channel gains are clamped so a near-empty
    channel (e.g. almost no red at depth) can't explode into noise."""
    result = bgr.astype(np.float32)
    means = result.reshape(-1, 3).mean(axis=0)          # per-channel mean (B, G, R)
    gray = float(means.mean())
    gains = np.clip(gray / (means + 1e-6), 0.4, 3.0)     # boost the attenuated channels
    result *= gains
    return np.clip(result, 0, 255).astype(np.uint8)


def _clahe_contrast(bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Local contrast on the lightness channel only, so colour is untouched."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance_bgr(bgr: np.ndarray, strength: float = DEFAULT_STRENGTH) -> np.ndarray:
    """Return an underwater-corrected copy of a BGR image.

    strength in [0, 1] blends corrected↔original (1.0 = full correction).
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    out = _clahe_contrast(_white_balance(bgr))
    if strength < 1.0:
        out = cv2.addWeighted(out, float(strength), bgr, 1.0 - float(strength), 0.0)
    return out
