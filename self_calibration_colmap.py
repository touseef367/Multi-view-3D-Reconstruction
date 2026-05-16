import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import pycolmap
except ImportError:
    sys.exit("pycolmap not found. Install with `pip install pycolmap` (requires COLMAP >=3.8).")

def run_cmd(cmd):
    """Run a shell command and abort on non‑zero return code."""
    print("[CMD]", " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        sys.exit(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Self calibrate from an image folder using COLMAP")
    parser.add_argument("--images_dir", type=Path, required=True, help="Folder with input images")
    parser.add_argument(
        "--working_dir",
        type=Path,
        required=True,
        help="Workspace to store COLMAP database, sparse & dense models",
    )
    parser.add_argument(
        "--single_camera",
        action="store_true",
        help="Force all images to share one intrinsic camera (recommended for fixed‑lens cameras)",
    )
    parser.add_argument(
        "--mapper_args",
        type=str,
        default="",
        help="Extra arguments forwarded verbatim to `colmap mapper`",
    )
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    work = args.working_dir.resolve()
    if not images_dir.is_dir():
        sys.exit(f"Images directory {images_dir} does not exist.")

    # Fresh workspace
    if work.exists():
        print(f"[INFO] Removing existing workspace {work}")
        shutil.rmtree(work)
    (work / "sparse").mkdir(parents=True)

    db_path = work / "database.db"

    # ------------------------------------------------------------------
    # 1. FEATURE EXTRACTION -------------------------------------------------
    # ------------------------------------------------------------------
    extract_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1" if args.single_camera else "0",
        # Use default SIFT parameters; adjust if you have low‑texture scenes
    ]
    run_cmd(extract_cmd)

    # ------------------------------------------------------------------
    # 2. FEATURE MATCHING ---------------------------------------------------
    # ------------------------------------------------------------------
    match_cmd = [
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        str(db_path),
    ]
    run_cmd(match_cmd)

    # ------------------------------------------------------------------
    # 3. INCREMENTAL MAPPING ------------------------------------------------
    # ------------------------------------------------------------------
    mapper_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(work / "sparse"),
    ] + args.mapper_args.split()
    run_cmd(mapper_cmd)

    # COLMAP writes one sub‑folder per model (n = 0 for the largest).
    model_dir = next((work / "sparse").glob("*"))
    print(f"[INFO] Using model in {model_dir}")

    # ------------------------------------------------------------------
    # 4. CONVERT TO TXT + LOAD WITH pycolmap --------------------------------
    # ------------------------------------------------------------------
    convert_cmd = [
        "colmap",
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(model_dir),
        "--output_type",
        "TXT",
    ]
    run_cmd(convert_cmd)

    rec = pycolmap.Reconstruction(str(model_dir))

    # ------------------------------------------------------------------
    # 5. Dump intrinsics & extrinsics ---------------------------------------
    # ------------------------------------------------------------------
    intrinsics = {}
    for cam_id, cam in rec.cameras.items():
        K = np.array(
            [
                [cam.fx, 0, cam.cx],
                [0, cam.fy, cam.cy],
                [0, 0, 1],
            ]
        )
        intrinsics[cam_id] = K
        print(f"Camera {cam_id}:\n{K}\n")

    extrinsics = {}
    for img_id, img in rec.images.items():
        R = img.R()
        t = img.tvec
        extrinsics[img.name] = {"R": R, "t": t}
        print(f"Image {img.name}: R =\n{R}\n t = {t}\n")

    # Optionally save numpy dumps
    np.save(work / "intrinsics.npy", intrinsics)
    np.save(work / "extrinsics.npy", extrinsics)

    # Also write a JSON for easy ingestion elsewhere
    with open(work / "camera_poses.json", "w", encoding="utf-8") as f:
        json.dump({k: {"R": v["R"].tolist(), "t": v["t"].tolist()} for k, v in extrinsics.items()}, f, indent=2)

    print("\n[✓] Self‑calibration complete. Results written to", work)


if __name__ == "__main__":
    main()
