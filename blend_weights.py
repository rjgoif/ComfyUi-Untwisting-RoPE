from __future__ import annotations

import re
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Weight parsing helpers
# ---------------------------------------------------------------------------

def _parse_weight_string(text: str) -> List[float]:
    """
    Parse a weight string into a list of positive floats.

    Accepts any mix of whitespace or punctuation as delimiters.
    Non-numeric tokens are silently skipped.
    Negative or zero values are clamped to a small positive epsilon.
    """
    tokens = re.split(r"[\s,;:|/\\]+", text.strip())
    weights: List[float] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            v = float(token)
            weights.append(max(1e-8, v))
        except ValueError:
            continue
    return weights


def resolve_blend_weights(
    weights_raw: Optional[List[float]],
    num_refs: int,
) -> List[float]:
    """
    Given a raw list of parsed weights and the number of reference streams,
    return a normalized list of length num_refs.

    Rules:
    - If weights_raw is None or empty: uniform weights.
    - If len(weights_raw) < num_refs: pad with the mean of the provided weights.
    - If len(weights_raw) > num_refs: truncate.
    - Normalize so weights sum to 1.0.
    """
    if num_refs <= 0:
        return []

    if not weights_raw:
        return [1.0 / num_refs] * num_refs

    weights = list(weights_raw)

    # Pad with mean if too short.
    if len(weights) < num_refs:
        mean_w = sum(weights) / len(weights)
        while len(weights) < num_refs:
            weights.append(mean_w)

    # Truncate if too long.
    weights = weights[:num_refs]

    # Normalize.
    total = sum(weights)
    if total <= 0:
        return [1.0 / num_refs] * num_refs

    return [w / total for w in weights]


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class RoPEBlendWeights:
    """
    Parses a user-supplied weight string into normalized blend weights for
    multi-reference UntwistingRoPE style transfer.

    Weight string format
    --------------------
    Enter one number per reference image, separated by any whitespace,
    commas, semicolons, or other punctuation. The numbers represent the
    relative influence of each reference image on the style transfer.

    Examples
    --------
    Equal weights (same as leaving blank):
        1 1 1

    First reference twice as influential:
        2 1 1

    Any delimiter works:
        2, 1; 1  |  2.0 1.0 1.0  |  2/1/1

    Mismatch handling
    -----------------
    - Fewer numbers than reference images: missing weights are filled with
      the mean of the provided numbers.
      Example: 4 images, weights "2 2 3 3" → padded to "2 2 3 3 2.5 2.5",
      then normalized.
    - More numbers than reference images: extra numbers are ignored.
    - Empty or blank: all references receive equal weight.

    All weights are normalized so they sum to 1.0. The individual values
    only set the relative influence of each reference — not the total
    style transfer strength (controlled by the UntwistingRoPE node).

    VRAM note
    ---------
    Each reference image requires its own RF inversion trajectory. Using
    N reference images costs approximately N times the RF inversion VRAM
    and compute of a single reference run.
    """

    CATEGORY = "model_patches/Untwisting RoPE"
    RETURN_TYPES = ("ROPE_BLEND_WEIGHTS",)
    RETURN_NAMES = ("blend_weights",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weights": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Relative weights for each reference image, separated by any "
                            "whitespace or punctuation (spaces, commas, semicolons, etc.).\n\n"
                            "Examples:\n"
                            "  Equal weight:      1 1 1\n"
                            "  First dominates:   3 1 1\n"
                            "  Any delimiter:     2, 1; 1\n\n"
                            "Fewer numbers than images → missing entries filled with mean of "
                            "provided numbers.\n"
                            "More numbers than images → extras ignored.\n"
                            "Empty → all references receive equal weight.\n\n"
                            "All weights are normalized to sum to 1.0.\n\n"
                            "VRAM note: each reference image requires its own RF inversion "
                            "trajectory, costing approximately N× the compute of a single "
                            "reference run."
                        ),
                    },
                ),
                "blend_mode": (
                    ["weighted_sum", "round_robin"],
                    {
                        "default": "round_robin",
                        "tooltip": (
                            "How reference images are combined at each attention block.\n\n"
                            "weighted_sum: all references are blended into one K/V at every "
                            "block, weighted by the values above. Good for stylistically "
                            "similar references.\n\n"
                            "round_robin: each block is assigned one reference by cycling "
                            "through them (block_index % N). Different references influence "
                            "different blocks. Better for stylistically different references "
                            "since they do not cancel each other out."
                        ),
                    },
                ),
            },
        }

    def build(self, weights: str = "", blend_mode: str = "round_robin") -> tuple:
        parsed = _parse_weight_string(weights) if weights.strip() else []
        # Store raw parsed weights and mode; resolution to num_refs happens at injection time.
        return ({"weights": parsed, "mode": blend_mode},)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "RoPEBlendWeights": RoPEBlendWeights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RoPEBlendWeights": "RoPE Blend Weights",
}

__all__ = [
    "RoPEBlendWeights",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "resolve_blend_weights",
    "_parse_weight_string",
]
