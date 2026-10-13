#!/usr/bin/env python3


#cd /home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META_BG_CDF/segm
#export PYTHONPATH="$(dirname "$PWD"):$PYTHONPATH"
#python3 video_infer.py --checkpoint /home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META_BG_CDF/segm/old_MODEL_FILE_0.7/checkpoint.pth --source /home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META_BG_CDF/segm/RRR2_BugFire_EC.mov --output /home/AD.UNLV.EDU/bhattb3/segmenter_SEMI_META_BG_CDF/segm/RRR2_BugFire_last2min_BG_CDF_fire_highlight.mp4 --seconds 120 --device cuda

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BG-CDF video inference.

Input:
    Video file (.mov, .mp4, etc.)

Output:
    MP4 with:
        LEFT  = original video
        RIGHT = fire class highlighted in red

Only the last N seconds are processed.

Fire class:
    1

Model:
    ViT-Tiny-Patch16-384 + Mask Transformer
    loaded from the supplied BG-CDF checkpoint.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Make repository importable
# -------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segm.model.factory import load_model
from segm.model.utils import inference


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

FIRE_CLASS = 1

WINDOW_SIZE = 512
WINDOW_STRIDE = 256

# Number of seconds from the end of the video to process.
DEFAULT_SECONDS = 120.0

# Batch size used INSIDE sliding-window inference.
# This is not the video-frame batch size.
INFERENCE_BATCH_SIZE = 1

# Overlay transparency.
FIRE_ALPHA = 0.50


# -------------------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BG-CDF fire segmentation on the last N seconds of a video."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to BG-CDF checkpoint.pth",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input video (.mov, .mp4, etc.)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output MP4 path",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="Number of seconds from the end of the video to process.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Device
# -------------------------------------------------------------------------

def get_device(device_string):
    if device_string.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_string)

    if device_string.startswith("cuda"):
        print("WARNING: CUDA requested but unavailable.")
        print("Falling back to CPU.")

    return torch.device("cpu")


# -------------------------------------------------------------------------
# Model input normalization
# -------------------------------------------------------------------------

def normalize_image(image_tensor, normalization_name):
    """
    Normalize image using the normalization convention used by the model.

    Segmenter commonly uses:
        vit:
            mean = [0.5, 0.5, 0.5]
            std  = [0.5, 0.5, 0.5]

    imagenet:
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
    """

    if normalization_name == "vit":
        mean = torch.tensor(
            [0.5, 0.5, 0.5],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.5, 0.5, 0.5],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        ).view(1, 3, 1, 1)

    elif normalization_name == "imagenet":
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        ).view(1, 3, 1, 1)

    else:
        print(
            f"WARNING: Unknown normalization '{normalization_name}'. "
            "Using raw [0,1] image values."
        )
        return image_tensor

    return (image_tensor - mean) / std


# -------------------------------------------------------------------------
# Fire overlay
# -------------------------------------------------------------------------

def create_fire_overlay(frame_bgr, segmentation):
    """
    Highlight only class FIRE_CLASS in red.

    Everything else remains visually unchanged.
    """

    fire_mask = segmentation == FIRE_CLASS

    output = frame_bgr.copy()

    # OpenCV uses BGR.
    # Red = (0, 0, 255)
    red = np.zeros_like(frame_bgr)
    red[:, :, 2] = 255

    output[fire_mask] = (
        FIRE_ALPHA * red[fire_mask]
        + (1.0 - FIRE_ALPHA) * frame_bgr[fire_mask]
    ).astype(np.uint8)

    return output, fire_mask


# -------------------------------------------------------------------------
# Side-by-side visualization
# -------------------------------------------------------------------------

def make_side_by_side(original, highlighted):
    """
    Place original and highlighted frames side-by-side.

    Both are 1920x1080 in this case, producing 3840x1080 output.
    """

    return np.concatenate(
        [original, highlighted],
        axis=1,
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()

    device = get_device(args.device)

    print("=" * 70)
    print("BG-CDF VIDEO FIRE INFERENCE")
    print("=" * 70)

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Input      : {source_path}")
    print(f"Output     : {output_path}")
    print(f"Device     : {device}")
    print(f"Fire class : {FIRE_CLASS}")
    print(f"Window     : {WINDOW_SIZE}")
    print(f"Stride     : {WINDOW_STRIDE}")
    print(f"Duration   : last {args.seconds:.1f} seconds")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------------
    # Check files
    # ---------------------------------------------------------------------

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Input video not found:\n{source_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------------

    print("Loading BG-CDF model...")

    model, variant = load_model(
        str(checkpoint_path)
    )

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print()

    # ---------------------------------------------------------------------
    # Read model configuration
    # ---------------------------------------------------------------------

    dataset_kwargs = variant.get(
        "dataset_kwargs",
        {},
    )

    normalization_name = dataset_kwargs.get(
        "normalization",
        "vit",
    )

    inference_kwargs = variant.get(
        "inference_kwargs",
        {},
    )

    # Use the trained model's configured inference parameters
    # when available.
    window_size = inference_kwargs.get(
        "window_size",
        WINDOW_SIZE,
    )

    window_stride = inference_kwargs.get(
        "window_stride",
        WINDOW_STRIDE,
    )

    print("Model inference configuration:")
    print(f"  Normalization : {normalization_name}")
    print(f"  Window size   : {window_size}")
    print(f"  Window stride : {window_stride}")
    print()

    # ---------------------------------------------------------------------
    # Open video
    # ---------------------------------------------------------------------

    cap = cv2.VideoCapture(
        str(source_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{source_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS."
        )

    duration = total_frames / fps

    print("Video information:")
    print(f"  Resolution : {frame_width} x {frame_height}")
    print(f"  FPS        : {fps:.3f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {duration:.2f} seconds")
    print()

    # ---------------------------------------------------------------------
    # Determine starting frame
    # ---------------------------------------------------------------------

    seconds_to_process = min(
        args.seconds,
        duration,
    )

    start_time = max(
        0.0,
        duration - seconds_to_process,
    )

    start_frame = int(
        round(start_time * fps)
    )

    frames_to_process = total_frames - start_frame

    print("Starting inference at:")
    print(f"  {start_time:.2f} seconds")
    print(f"  frame {start_frame}")
    print()

    print("Running frame-by-frame inference...")
    print("Press Ctrl+C to stop.")
    print()

    # ---------------------------------------------------------------------
    # Seek to beginning of last N seconds
    # ---------------------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    # ---------------------------------------------------------------------
    # Create output writer
    # ---------------------------------------------------------------------

    output_width = frame_width * 2
    output_height = frame_height

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            output_width,
            output_height,
        ),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    # ---------------------------------------------------------------------
    # Processing statistics
    # ---------------------------------------------------------------------

    processed = 0
    total_fire_pixels = 0
    total_pixels = 0

    start_wall_time = time.time()

    # ---------------------------------------------------------------------
    # Frame loop
    # ---------------------------------------------------------------------

    try:

        with torch.no_grad():

            while True:

                ret, frame_bgr = cap.read()

                if not ret:
                    break

                # ---------------------------------------------------------
                # Convert BGR -> RGB
                # ---------------------------------------------------------

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                # ---------------------------------------------------------
                # Convert image to PyTorch tensor
                #
                # Shape:
                #     [1, 3, H, W]
                #
                # Values:
                #     [0, 1]
                # ---------------------------------------------------------

                image_tensor = torch.from_numpy(
                    frame_rgb.astype(
                        np.float32
                    ) / 255.0
                )

                image_tensor = image_tensor.permute(
                    2,
                    0,
                    1,
                )

                image_tensor = image_tensor.unsqueeze(
                    0
                )

                image_tensor = image_tensor.to(
                    device,
                    non_blocking=True,
                )

                # ---------------------------------------------------------
                # Apply the same normalization expected by the model.
                # ---------------------------------------------------------

                image_tensor = normalize_image(
                    image_tensor,
                    normalization_name,
                )

                # ---------------------------------------------------------
                # Metadata expected by the repository's inference()
                #
                # The actual Segmenter inference implementation expects:
                #
                # inference(
                #     model,
                #     ims,
                #     ims_metas,
                #     ori_shape,
                #     window_size,
                #     window_stride,
                #     batch_size,
                # )
                # ---------------------------------------------------------

                ims = [
                    image_tensor
                ]

                ims_metas = [
                    {
                        "flip": False
                    }
                ]

                ori_shape = (
                    frame_height,
                    frame_width,
                )

                # ---------------------------------------------------------
                # Run repository sliding-window inference
                # ---------------------------------------------------------

                logits = inference(
                    model,
                    ims,
                    ims_metas,
                    ori_shape,
                    window_size,
                    window_stride,
                    INFERENCE_BATCH_SIZE,
                )

                # ---------------------------------------------------------
                # Repository inference returns:
                #
                #     [num_classes, H, W]
                #
                # Convert to class prediction.
                # ---------------------------------------------------------

                if logits.dim() == 4:
                    logits = logits.squeeze(0)

                prediction = logits.argmax(
                    dim=0
                )

                prediction = prediction.detach().cpu().numpy()

                # ---------------------------------------------------------
                # Safety resize if necessary.
                # ---------------------------------------------------------

                if prediction.shape != (
                    frame_height,
                    frame_width,
                ):
                    prediction = cv2.resize(
                        prediction.astype(np.uint8),
                        (
                            frame_width,
                            frame_height,
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # ---------------------------------------------------------
                # Create red fire overlay
                # ---------------------------------------------------------

                highlighted_frame, fire_mask = (
                    create_fire_overlay(
                        frame_bgr,
                        prediction,
                    )
                )

                # ---------------------------------------------------------
                # Side-by-side:
                #
                # LEFT  = original
                # RIGHT = BG-CDF fire prediction
                # ---------------------------------------------------------

                output_frame = make_side_by_side(
                    frame_bgr,
                    highlighted_frame,
                )

                writer.write(
                    output_frame
                )

                # ---------------------------------------------------------
                # Statistics
                # ---------------------------------------------------------

                fire_pixels = int(
                    fire_mask.sum()
                )

                total_fire_pixels += fire_pixels
                total_pixels += fire_mask.size

                processed += 1

                # ---------------------------------------------------------
                # Progress display
                # ---------------------------------------------------------

                if processed == 1 or processed % 24 == 0:

                    elapsed = (
                        time.time()
                        - start_wall_time
                    )

                    fps_processing = (
                        processed / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    progress = (
                        processed
                        / max(frames_to_process, 1)
                    )

                    fire_percent = (
                        100.0
                        * fire_pixels
                        / fire_mask.size
                    )

                    avg_fire_percent = (
                        100.0
                        * total_fire_pixels
                        / max(total_pixels, 1)
                    )

                    print(
                        f"\r"
                        f"Progress: {progress * 100:6.2f}% | "
                        f"Frame: {processed:6d}/"
                        f"{frames_to_process:6d} | "
                        f"Processing FPS: {fps_processing:5.2f} | "
                        f"Fire: {fire_percent:6.2f}% | "
                        f"Avg Fire: {avg_fire_percent:6.2f}%",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:

        print()
        print()
        print("Inference interrupted by user.")

    finally:

        cap.release()
        writer.release()

    # ---------------------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------------------

    elapsed = (
        time.time()
        - start_wall_time
    )

    average_fire_percent = (
        100.0
        * total_fire_pixels
        / max(total_pixels, 1)
    )

    print()
    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Frames processed     : {processed}")
    print(f"Processing time      : {elapsed:.2f} seconds")

    if elapsed > 0:
        print(
            f"Average processing  : "
            f"{processed / elapsed:.2f} frames/sec"
        )

    print(
        f"Average fire pixels : "
        f"{average_fire_percent:.2f}%"
    )

    print()
    print("Output video:")
    print(output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()