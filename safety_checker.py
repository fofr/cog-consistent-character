import torch
import os
import subprocess
import numpy as np
import PIL
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor

FEATURE_EXTRACTOR = "./feature-extractor"
SAFETY_CACHE = "./safety-cache"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


class SafetyChecker:
    def __init__(self):
        if not os.path.exists(SAFETY_CACHE):
            subprocess.check_call(
                ["pget", "-xf", SAFETY_URL, SAFETY_CACHE],
                close_fds=False,
            )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

    def load_image(self, image_path):
        return PIL.Image.open(image_path).convert("RGB")

    def run(self, image_paths, error_on_all_nsfw=True):
        images = [self.load_image(image_path) for image_path in image_paths]
        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(
            "cuda"
        )
        np_images = [np.array(val) for val in images]
        _, is_nsfw = self.safety_checker(
            images=np_images,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )

        for i, nsfw in enumerate(is_nsfw):
            if nsfw:
                print(f"NSFW content detected in image {image_paths[i]}")

        if error_on_all_nsfw and all(is_nsfw):
            raise Exception(
                "NSFW content detected in all outputs. Try running it again, or try a different prompt."
            )

        return is_nsfw
