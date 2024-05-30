# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import random
from PIL import Image, ExifTags
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

MAX_HEADSHOTS = 59
MAX_POSES = 30
POSE_PATH = f"{INPUT_DIR}/poses"

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

        # Download pose images
        self.comfyUI.weights_downloader.download(
            "pose_images.tar",
            "https://weights.replicate.delivery/default/fofr/character/pose_images.tar",
            f"{INPUT_DIR}/poses",
        )

        self.headshots = self.list_pose_filenames(type="headshot")
        self.poses = self.list_pose_filenames(type="pose")
        self.all = self.headshots + self.poses

    def get_filenames(
        self, filenames: List[str], length: int, use_random: bool = True
    ) -> List[str]:
        if length > len(filenames):
            length = len(filenames)
            print(f"Using {length} as the max number of files.")

        if use_random:
            random.shuffle(filenames)

        return filenames[:length]

    def list_pose_filenames(self, type="headshot"):
        if type == "headshot":
            max_value = MAX_HEADSHOTS
            prefix = "headshot"
        elif type == "pose":
            max_value = MAX_POSES
            prefix = "pose"
        else:
            raise ValueError("Invalid type specified. Use 'headshot' or 'pose'.")

        return [
            {
                "kps": f"{POSE_PATH}/{prefix}_kps_{str(i).zfill(5)}_.png",
                "openpose": f"{POSE_PATH}/{prefix}_open_pose_{str(i).zfill(5)}_.png",
                "dwpose": f"{POSE_PATH}/{prefix}_dw_pose_{str(i).zfill(5)}_.png",
            }
            for i in range(1, max_value + 1)
        ]

    def get_poses(self, number_of_outputs, is_random, type):
        if type == "Headshot poses":
            return self.get_filenames(self.headshots, number_of_outputs, is_random)
        elif type == "Half-body poses":
            return self.get_filenames(self.poses, number_of_outputs, is_random)
        else:
            return self.get_filenames(self.all, number_of_outputs, is_random)

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
        check_orientation: bool = True,
    ):
        image = Image.open(input_file)

        if check_orientation:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = dict(image._getexif().items())

                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (KeyError, AttributeError):
                # EXIF data does not have orientation
                # Do not rotate
                pass

        image.save(os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        positive_prompt = workflow["9"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["10"]["inputs"]
        negative_prompt["text"] = f"nsfw, naked, nude, {kwargs['negative_prompt']}"

        sampler = workflow["11"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        empty_latent_image = workflow["29"]["inputs"]
        empty_latent_image["batch_size"] = kwargs["number_of_images_per_pose"]

        kps_input_image = workflow["94"]["inputs"]
        kps_input_image["image"] = kwargs["pose"]["kps"]

        dwpose_input_image = workflow["95"]["inputs"]
        dwpose_input_image["image"] = kwargs["pose"]["dwpose"]

    def predict(
        self,
        prompt: str = Input(
            default="A headshot photo",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        subject: Path = Input(
            description="An image of a person. Best images are square close ups of a face, but they do not have to be.",
            default=None,
        ),
        type: str = Input(
            description="The type of images to generate, headshots, half-body poses or both.",
            choices=[
                "Both headshots and half-body poses",
                "Headshot poses",
                "Half-body poses",
            ],
            default="Both headshots and half-body poses",
        ),
        number_of_outputs: int = Input(
            description="The number of images to generate.", default=5, ge=1, le=90
        ),
        number_of_images_per_pose: int = Input(
            description="The number of images to generate for each pose.",
            default=1,
            ge=1,
            le=4,
        ),
        randomise_poses: bool = Input(
            description="Randomise the poses used.", default=True
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        using_fixed_seed = bool(seed)
        seed = seed_helper.generate(seed)

        self.handle_input_file(subject, filename="subject.png")
        poses = self.get_poses(number_of_outputs, randomise_poses, type)

        print(poses)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.comfyUI.connect()

        if using_fixed_seed:
            self.comfyUI.reset_execution_cache()

        for pose in poses:
            self.update_workflow(
                workflow,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                type=type,
                number_of_outputs=number_of_outputs,
                number_of_images_per_pose=number_of_images_per_pose,
                randomise_poses=randomise_poses,
                pose=pose,
            )
            self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
