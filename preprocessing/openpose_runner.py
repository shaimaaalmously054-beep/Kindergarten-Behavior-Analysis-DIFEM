"""
OpenPose Skeleton Extraction Script
----------------------------------
This script is used to extract skeletal keypoints from video frames
as a preprocessing step before applying the DIFEM method.

Note:
- This script does NOT include datasets.
- This script is NOT part of the behavior decision logic.
- OpenPose must be installed separately following the official instructions:
  https://github.com/CMU-Perceptual-Computing-Lab/openpose

Author: Shaimaa Almously
Graduation Project – 2025
"""

import os
import subprocess
import re
from pathlib import Path


def extract_number(name):
    numbers = re.findall(r'\d+', name)
    return int(numbers[0]) if numbers else -1


def run_openpose_on_frames(
    base_input_dir,
    base_json_output_dir,
    base_image_output_dir,
    batch_size=100
):
    """
    Runs OpenPose on extracted video frames to generate skeleton JSON files.

    Args:
        base_input_dir (str): Directory containing frame folders.
        base_json_output_dir (str): Output directory for JSON keypoints.
        base_image_output_dir (str): Output directory for rendered images.
        batch_size (int): Number of videos processed per batch.
    """

    categories = ["Category_A"]  # Generic naming to avoid dataset leakage

    for category in categories:
        input_dir = os.path.join(base_input_dir, category)
        json_output_dir = os.path.join(base_json_output_dir, category)
        image_output_dir = os.path.join(base_image_output_dir, category)

        os.makedirs(json_output_dir, exist_ok=True)
        os.makedirs(image_output_dir, exist_ok=True)

        video_dirs = sorted(
            [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))],
            key=extract_number
        )

        for batch_start in range(0, len(video_dirs), batch_size):
            batch_dirs = video_dirs[batch_start:batch_start + batch_size]

            for video_dir in batch_dirs:
                frames_dir = os.path.join(input_dir, video_dir)
                video_json_output = os.path.join(json_output_dir, video_dir)
                video_image_output = os.path.join(image_output_dir, video_dir)

                os.makedirs(video_json_output, exist_ok=True)
                os.makedirs(video_image_output, exist_ok=True)

                if any(f.endswith(".json") for f in os.listdir(video_json_output)):
                    continue

                command = (
                    f'bin\\OpenPoseDemo.exe '
                    f'--image_dir "{frames_dir}" '
                    f'--write_json "{video_json_output}" '
                    f'--write_images "{video_image_output}" '
                    f'--display 0 '
                    f'--render_pose 1 '
                    f'--number_people_max 3 '
                    f'--model_pose BODY_25 '
                    f'--scale_number 4 '
                    f'--scale_gap 0.25 '
                )

                subprocess.run(command, shell=True)


if __name__ == "__main__":
    # Example usage (paths must be adapted by the user)
    run_openpose_on_frames(
        base_input_dir="path/to/frames",
        base_json_output_dir="path/to/json_output",
        base_image_output_dir="path/to/image_output"
    )
