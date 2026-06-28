from __future__ import annotations

"""MultiRefEncode node for ComfyUi-Untwisting-RoPE.

Takes a VAE and a batch of images [N, H, W, C], encodes each image
separately through the VAE, then concatenates the resulting latents
along dim 0 to produce a true [N, ...] latent batch.

This is necessary because some VAEs (e.g. qwen_image_vae) do not
respect the batch dimension when encoding multiple images at once —
they collapse the batch into a single latent. Encoding separately
and concatenating guarantees each reference image gets its own
independent latent encoding.

Outputs:
  latents  - LATENT dict with samples [N, C, ...] batched along dim 0
  count    - INT, number of reference images encoded (= N)
"""

import torch
from typing import Any, Dict, Tuple


class MultiRefEncode:
    """Encode a batch of images into separate latents and concatenate them.

    Connect an IMAGE batch (e.g. from ImageBatchMulti or MakeBatch) and
    your VAE. Each image in the batch is encoded independently, then the
    resulting latents are concatenated along the batch dimension.

    Use the output LATENT directly as the reference_latent input of the
    RFInversion node for multi-reference style transfer.

    The count output tells you how many reference streams are in the
    latent batch — useful for wiring up the RoPE Blend Weights node.
    """

    CATEGORY = "model_patches/Untwisting RoPE"
    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latents", "count")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "A batch of reference images [N, H, W, C]. "
                            "Each image is encoded separately so that VAEs "
                            "which collapse batch dimensions (e.g. qwen_image_vae) "
                            "still produce N independent latents."
                        ),
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": "VAE used to encode the reference images.",
                    },
                ),
            },
        }

    def encode(self, images: torch.Tensor, vae: Any) -> Tuple[Dict[str, Any], int]:
        """Encode each image in the batch separately and concatenate latents.

        Args:
            images: [N, H, W, C] image batch.
            vae:    ComfyUI VAE object with an .encode() method.

        Returns:
            Tuple of (latent_dict, count) where latent_dict["samples"]
            has shape [N, C, ...] with all reference latents batched
            along dim 0.
        """
        if not torch.is_tensor(images) or images.ndim != 4:
            raise RuntimeError(
                f"MultiRefEncode expected images as [N, H, W, C] (4D tensor), "
                f"got shape {tuple(images.shape) if torch.is_tensor(images) else type(images).__name__}."
            )

        n = int(images.shape[0])
        if n < 1:
            raise RuntimeError("MultiRefEncode received an empty image batch.")

        encoded_latents = []
        batch_indices = []

        for i in range(n):
            # Extract single image and add batch dim back: [1, H, W, C]
            img_i = images[i:i + 1]

            # Encode through VAE — result is a latent dict or tensor.
            latent_i = vae.encode(img_i)

            # Normalise to tensor.
            if isinstance(latent_i, dict):
                samples_i = latent_i["samples"]
            elif torch.is_tensor(latent_i):
                samples_i = latent_i
            else:
                raise RuntimeError(
                    f"MultiRefEncode: VAE encode returned unexpected type "
                    f"{type(latent_i).__name__} for image {i}."
                )

            encoded_latents.append(samples_i)
            batch_indices.extend(list(range(int(samples_i.shape[0]))))

        # Validate that all latents share the same non-batch shape.
        ref_shape = encoded_latents[0].shape[1:]
        for i, lat in enumerate(encoded_latents):
            if lat.shape[1:] != ref_shape:
                raise RuntimeError(
                    f"MultiRefEncode: latent shape mismatch between image 0 "
                    f"({tuple(encoded_latents[0].shape)}) and image {i} "
                    f"({tuple(lat.shape)}). All reference images must have the "
                    f"same spatial dimensions after encoding."
                )

        # Concatenate along batch dim 0.
        samples_out = torch.cat(encoded_latents, dim=0)

        latent_out = {
            "samples": samples_out,
            "batch_index": batch_indices,
        }

        return (latent_out, n)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MultiRefEncode": MultiRefEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiRefEncode": "Multi-Ref Encode",
}

__all__ = [
    "MultiRefEncode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
