# 3D Reconstruction (COLMAP + OpenCV + Open3D)

It implements a classical multi-view 3D reconstruction workflow:

1. **Camera calibration / pose estimation (SfM)** with COLMAP (intrinsics + extrinsics).
2. **Stereo rectification** for selected image pairs.
3. **Dense matching** (disparity) using OpenCV StereoSGBM.
4. **Depth / point cloud generation** by reprojecting disparity to 3D.
5. **Point cloud fusion** (concatenate + downsample + outlier removal) and export.

The main entry point for reconstruction is `reconstruction.py`.

## Data source (ETH3D)

The COLMAP TXT model files and the images can be acquired from the ETH3D dataset website:

https://www.eth3d.net/datasets

You can use ETH3D (e.g., the **Office** scene) or your own images. If you use ETH3D, download the dataset from the link above.

This repo contain example data under `data/`:

- the example images under `data/images/`
- the matching COLMAP TXT model files under `data/` (`cameras.txt`, `images.txt`, `points3D.txt`)


## Repository contents

- `reconstruction.py` (main):
   - Loads COLMAP TXT files (`data/cameras.txt`, `data/images.txt`, `data/points3D.txt`)
   - Loads the corresponding images from `data/images/`
   - Rectifies pairs, computes disparity, reprojects to 3D, writes per-pair point clouds
   - Fuses all generated point clouds and saves a final `fused.ply`

- `self_calibration_colmap.py` (optional utility):
   - Runs the COLMAP CLI pipeline on an image folder to self-calibrate (estimate intrinsics/extrinsics)
   - Useful if you want to generate your own COLMAP model instead of downloading the provided TXT files

## Expected folder layout

```
data/
   cameras.txt
   images.txt
   points3D.txt
   images/
      DSC_0249.JPG
      DSC_0250.JPG
      ...
```

The reconstruction script uses only images that physically exist in `data/images/`. It matches poses by **basename** (so it tolerates COLMAP image names like `some/folder/DSC_0250.JPG`).

## How the reconstruction works (practical implementation)

### 1) Load COLMAP calibration

`reconstruction.py` starts by parsing:

- `data/cameras.txt` → intrinsics matrix $K$ (supports common COLMAP models like `PINHOLE`, `SIMPLE_PINHOLE`, `RADIAL`, …)
- `data/images.txt` → camera pose per image (quaternion + translation)

### 2) Choose image pairs

The script sorts the available image filenames and processes consecutive pairs (up to the `pairs=...` argument in the call).

### 3) Epipolar geometry and stereo rectification

For each pair:

- Computes the relative pose $R_{rel}, T_{rel}$ between the two COLMAP camera poses.
- Builds an essential/fundamental matrix and visualizes epipolar lines.
- Rectifies the pair using OpenCV (`stereoRectify`, `initUndistortRectifyMap`, `remap`).

Rectified images are saved to:

`data/output/rectified/`

### 4) Disparity (StereoSGBM)

- Converts rectified images to grayscale.
- Applies CLAHE and denoising to help matching.
- Computes disparity with StereoSGBM.

Disparity previews are saved to:

`data/output/disparities/`

### 5) Point cloud per pair

- Uses `reprojectImageTo3D(disp, Q)` to back-project disparity into a 3D point per pixel.
- Filters invalid points (disparity > 0 and finite depth).
- Colors points using the rectified left image.

Per-pair point clouds are saved to:

`data/output/pointclouds/cloud_*.ply`

### 6) Fusion

All pair clouds are concatenated and post-processed:

- voxel downsample
- statistical outlier removal
- normal estimation (useful if you later mesh)

Final output:

`data/output/fused.ply`

## Install

```bash
pip install -r requirements.txt
```

Optional (only needed for `self_calibration_colmap.py`):

- COLMAP installed and available as `colmap` on your PATH
- `pycolmap` installed and compatible with your COLMAP version/build

## How to run

### A) Run the main reconstruction

1. Ensure your images are in `data/images/`.
2. Ensure the matching COLMAP TXT model files are in `data/`:
   - `cameras.txt` (intrinsics)
   - `images.txt` (poses/extrinsics)
   - `points3D.txt` (3D points)

The TXT model must match your images (the image names in `images.txt` must match the filenames in `data/images/` by basename).

Run:

```bash
python reconstruction.py
```

### B) Generate calibration with the self-calibration script (COLMAP)

This is an alternative to downloading COLMAP TXT files.

```bash
python self_calibration_colmap.py --images_dir data/images --working_dir colmap_workspace --single_camera
```

This will run COLMAP, convert the model to TXT, and export:

- `colmap_workspace/intrinsics.npy`
- `colmap_workspace/extrinsics.npy`
- `colmap_workspace/camera_poses.json`

If you want to use those results with `reconstruction.py`, you should also ensure you have the COLMAP TXT files in `data/` (e.g., by copying/exporting `cameras.txt`, `images.txt`, `points3D.txt` from the COLMAP model converter output).

Tip: `reconstruction.py` expects those TXT files specifically under `data/`.

