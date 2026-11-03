"""Extract I3D features from a quadrant of side-by-side videos for FVD.

Reads video_*_side_by_side.mp4 files (1280x704), crops to one of four quadrants
(GT/Gen x Bot1/Bot2), extracts I3D features (400-dim logits), and saves
features + mu/sigma for FVD computation.

Uses the standard TF-Hub I3D (InceptionV1, Kinetics-400) ported to TorchScript
by the StyleGAN-V authors — the canonical model for FVD.

Quadrant layout (1280x704 -> four 640x352 crops):
  0 = GT  Bot1  (top-left)      1 = Gen Bot1 (top-right)
  2 = GT  Bot2  (bottom-left)   3 = Gen Bot2 (bottom-right)
"""

import argparse
import pathlib
import pprint
import re
import shutil
import time

import av
import huggingface_hub
import numpy as np
import torch
import tqdm

# (name, row_slice, col_slice) — each quadrant is 640x352
QUADRANTS: dict[int, tuple[str, slice, slice]] = {
    0: ("gt_bot1",  slice(0, 352), slice(0, 640)),
    1: ("gen_bot1", slice(0, 352), slice(640, 1280)),
    2: ("gt_bot2",  slice(352, 704), slice(0, 640)),
    3: ("gen_bot2", slice(352, 704), slice(640, 1280)),
}


def discover_videos(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Glob video_*_side_by_side.mp4 and sort by numeric index.

    Args:
        data_dir: Directory containing side-by-side mp4 files.

    Returns:
        Sorted list of video paths.
    """
    videos = list(data_dir.glob("video_*_side_by_side.mp4"))
    assert len(videos) > 0, f"No video_*_side_by_side.mp4 found in {data_dir}"
    videos.sort(key=lambda p: int(re.search(r"video_(\d+)_", p.name).group(1)))
    return videos


def decode_and_crop(
    video_path: pathlib.Path,
    row_slice: slice,
    col_slice: slice,
) -> np.ndarray:
    """Decode video and crop each frame to the specified quadrant.

    Args:
        video_path: Path to the mp4 file.
        row_slice: Row slice for quadrant crop.
        col_slice: Column slice for quadrant crop.

    Returns:
        [T, H_crop, W_crop, 3] uint8 numpy array (H_crop=352, W_crop=640).
    """
    container = av.open(str(video_path))
    frames = [
        frame.to_ndarray(format="rgb24")[row_slice, col_slice]
        for frame in container.decode(video=0)
    ]
    container.close()
    return np.stack(frames)  # [T, 352, 640, 3]


def preprocess_video(
    frames: np.ndarray,
    num_frames: int,
    resolution: int,
) -> torch.Tensor:
    """Optionally subsample, resize, and normalize a video clip for I3D.

    Args:
        frames: [T, H, W, 3] uint8 array.
        num_frames: If >0, uniformly sample this many frames. If 0, use all.
        resolution: Square spatial resolution for I3D (224).

    Returns:
        [3, T, resolution, resolution] float32 tensor in [-1, 1].
    """
    if num_frames > 0:
        indices = np.linspace(0, frames.shape[0] - 1, num=num_frames, dtype=int)
        frames = frames[indices]

    # [T, H, W, 3] -> [T, 3, H, W] float32 in [0, 1]
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    # Resize to [T, 3, res, res]
    tensor = torch.nn.functional.interpolate(
        tensor, size=(resolution, resolution), mode="bilinear", align_corners=False,
    )

    # Normalize to [-1, 1], rearrange to I3D layout [3, T, res, res]
    tensor = (2.0 * tensor - 1.0).permute(1, 0, 2, 3)
    return tensor


def load_i3d_model(
    weights_path: pathlib.Path,
    device: torch.device,
) -> torch.jit.ScriptModule:
    """Load I3D TorchScript model, downloading from HuggingFace if absent.

    Args:
        weights_path: Local path to save/load i3d_torchscript.pt.
        device: Torch device to load the model onto.

    Returns:
        I3D TorchScript model in eval mode on the given device.
    """
    if not weights_path.exists():
        print(f"Downloading I3D weights to {weights_path} ...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        cached = huggingface_hub.hf_hub_download(
            repo_id="flateon/FVD-I3D-torchscript",
            filename="i3d_torchscript.pt",
        )
        shutil.copy2(src=cached, dst=weights_path)
        print(f"Downloaded ({weights_path.stat().st_size / 1e6:.1f} MB)")
    return torch.jit.load(str(weights_path), map_location=device).eval()


def compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of feature vectors.

    Args:
        features: [N, D] float array.

    Returns:
        (mu [D], sigma [D, D]) both as float64.
    """
    mu = np.mean(features, axis=0).astype(np.float64)
    sigma = np.cov(features, rowvar=False).astype(np.float64)
    return mu, sigma


def main() -> None:
    """Extract I3D features from a quadrant of side-by-side videos."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with video_*_side_by_side.mp4 files")
    parser.add_argument("--quadrant", type=int, choices=[0, 1, 2, 3], required=True,
                        help="0=GT Bot1 (TL), 1=Gen Bot1 (TR), "
                             "2=GT Bot2 (BL), 3=Gen Bot2 (BR)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path")
    parser.add_argument("--i3d_weights", type=str, default="models/i3d_torchscript.pt",
                        help="Path to I3D TorchScript weights (auto-downloads if missing)")
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Frames to uniformly sample per video (0 = all)")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Spatial resolution for I3D input")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for I3D inference")
    args = parser.parse_args()

    quad_name, row_slice, col_slice = QUADRANTS[args.quadrant]

    print("=" * 60)
    print("FVD Feature Extraction (side-by-side quadrant)")
    print("=" * 60)
    pprint.pprint(vars(args))
    print(f"Quadrant {args.quadrant}: {quad_name} "
          f"(rows {row_slice.start}:{row_slice.stop}, "
          f"cols {col_slice.start}:{col_slice.stop})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_dir = pathlib.Path(args.data_dir)
    output_path = pathlib.Path(args.output)

    # Discover videos
    videos = discover_videos(data_dir=data_dir)
    print(f"\n{len(videos)} videos discovered")

    # Peek at first video
    with av.open(str(videos[0])) as c:
        s = c.streams.video[0]
        print(f"Video: {s.width}x{s.height}, codec={s.codec_context.name}")
    print(f"Crop: {col_slice.stop - col_slice.start}x{row_slice.stop - row_slice.start}")

    # Load I3D
    print(f"\nLoading I3D from {args.i3d_weights} ...")
    t0 = time.time()
    model = load_i3d_model(
        weights_path=pathlib.Path(args.i3d_weights),
        device=device,
    )
    print(f"Loaded in {time.time() - t0:.1f}s")

    frames_desc = "all" if args.num_frames == 0 else str(args.num_frames)
    print(f"\nExtracting {quad_name} features, {frames_desc} frames/video ...")

    # Feature extraction
    all_features: np.ndarray | None = None
    t_start = time.time()

    for batch_off in tqdm.trange(0, len(videos), args.batch_size, desc="Videos"):
        batch_paths = videos[batch_off : batch_off + args.batch_size]
        batch_tensors = []

        for vpath in batch_paths:
            frames = decode_and_crop(
                video_path=vpath, row_slice=row_slice, col_slice=col_slice,
            )  # [T, 352, 640, 3]
            tensor = preprocess_video(
                frames=frames, num_frames=args.num_frames, resolution=args.resolution,
            )  # [3, T, 224, 224]
            batch_tensors.append(tensor)
            del frames

        batch_tensor = torch.stack(batch_tensors).to(device=device)  # [B, 3, T, 224, 224]
        del batch_tensors

        with torch.inference_mode():
            feats = model(
                batch_tensor, rescale=False, resize=False, return_features=True,
            )  # [B, 400]

        feats_np = feats.cpu().numpy()
        del batch_tensor

        if all_features is None:
            feat_dim = feats_np.shape[1]
            all_features = np.zeros((len(videos), feat_dim), dtype=np.float32)
            print(f"Feature dim: {feat_dim}")

        all_features[batch_off : batch_off + len(batch_paths)] = feats_np

    elapsed = time.time() - t_start
    assert all_features is not None, "No videos processed"
    print(f"\n{len(videos)} videos in {elapsed:.1f}s ({len(videos) / elapsed:.1f} vid/s)")
    print(f"Features shape: {all_features.shape}")

    # Compute statistics
    mu, sigma = compute_statistics(features=all_features)
    print(f"mu: shape={mu.shape}, ||mu||={np.linalg.norm(mu):.4f}")
    print(f"sigma: shape={sigma.shape}, trace={np.trace(sigma):.4f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, features=all_features, mu=mu, sigma=sigma)
    print(f"\nSaved -> {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
