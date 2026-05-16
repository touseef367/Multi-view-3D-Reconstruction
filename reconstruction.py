import numpy as np
import cv2 as cv
from pathlib import Path
import open3d as o3d
import random
from typing import Optional

def load_cameras(cameras_txt: Path):
    """
    Reads camera intrinsics from COLMAP’s cameras.txt.
    Returns a dict: cam_id → {K: 3×3, dist: (k1,k2,p1,p2), width, height}
    """
    cams = {}
    with open(cameras_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            cam_id, model = int(tokens[0]), tokens[1]
            w, h = map(int, tokens[2:4])
            params = list(map(float, tokens[4:]))
            # Handle the common COLMAP models:
            if model == 'PINHOLE':
                fx, fy, cx, cy = params[:4]
                dist = np.zeros(4)
            elif model == 'SIMPLE_PINHOLE':
                fx = fy = params[0]
                cx, cy = params[1:3]
                dist = np.zeros(4)
            elif model == 'SIMPLE_RADIAL':
                fx = fy = params[0]
                cx, cy = params[1:3]
                k1 = params[3]
                dist = np.array([k1, 0, 0, 0])
            elif model == 'RADIAL':
                fx = fy = params[0]
                cx, cy = params[1:3]
                k1, k2 = params[3:5]
                dist = np.array([k1, k2, 0, 0])
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            K = np.array([[fx, 0, cx],
                          [ 0, fy, cy],
                          [ 0,  0,  1]])
            cams[cam_id] = {'K': K, 'dist': dist, 'width': w, 'height': h}
    return cams

def load_image_poses(images_txt: Path):
    """
    Reads extrinsics from COLMAP’s images.txt.
    Returns a list of dicts: [{name, cam_id, R:3×3, t:3×1}, …]
    """
    imgs = []
    with open(images_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # only header lines have exactly 10 tokens
            if len(parts) != 10:
                continue
            image_id     = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz    = map(float, parts[5:8])
            cam_id        = int(parts[8])
            name          = parts[9]
            # quaternion → rotation matrix
            w_, x_, y_, z_ = qw, qx, qy, qz
            R = np.array([
                [1-2*(y_*y_+z_*z_), 2*(x_*y_-z_*w_),   2*(x_*z_+y_*w_)],
                [2*(x_*y_+z_*w_),   1-2*(x_*x_+z_*z_), 2*(y_*z_-x_*w_)],
                [2*(x_*z_-y_*w_),   2*(y_*z_+x_*w_),   1-2*(x_*x_+y_*y_)]
            ])
            t = np.array([[tx], [ty], [tz]], dtype=float)
            imgs.append({'image_id': image_id,
                         'name': name,
                         'cam_id': cam_id,
                         'R': R,
                         't': t})
    return imgs

def draw_epilines_for_keypoints(imgL, imgR, F, pts, color=(0,0,255), thickness=10):
    """
    imgL: left BGR image
    imgR: right BGR image
    F:    3×3 fundamental matrix (uncalibrated)
    pts:  N×2 array of (x,y) in left image
    """
    outL = imgL.copy()
    outR = imgR.copy()
    # draw points on left
    for (x,y) in pts.astype(int):
        cv.circle(outL, (x,y), 8, color, thickness)

    line_thickness = 3   
    # compute epilines in right for pts in left
    linesR = cv.computeCorrespondEpilines(pts.reshape(-1,1,2), 1, F).reshape(-1,3)
    h, w = imgR.shape[:2]
    for (a,b,c) in linesR:
        # draw line a x + b y + c = 0
        # y0 = –(c + a*0)/b, y1 = –(c + a*w)/b
        y0 = int(-c/b)
        y1 = int(-(c + a*w)/b)
        cv.line(outR, (0,y0), (w,y1), color, line_thickness)
    return outL, outR

def draw_horizontal_for_keypoints(rL, rR, ptsL_rect, thickness=3):
    """
    rL, rR: rectified left/right BGR images
    ptsL_rect: N×2 array of (x,y) in rectified left image
    """
    outL = rL.copy()
    outR = rR.copy()
    h, w = rL.shape[:2]
    # draw points and horizontal lines
    for (x,y) in ptsL_rect.astype(int):
        cv.circle(outL, (x,y), 8, (0,0,255), thickness)
        # a horizontal line at y across both images
        cv.line(outL, (0,y), (w,y), (0,0,255), thickness)
        cv.line(outR, (0,y), (w,y), (0,0,255), thickness)
    return outL, outR


def compute_show_save_depth(disp: np.ndarray,
                            Q: np.ndarray,
                            out_dir: Path,
                            tag: str):
    """
    disp: float32 disparity map (H×W)
    Q:    4×4 reprojection matrix from stereoRectify
    Saves depth_{tag}.png and shows it.
    """
    # Reproject to 3D and extract the Z (depth) channel
    points3D = cv.reprojectImageTo3D(disp, Q)    # H×W×3
    depth_map = points3D[:, :, 2]                # H×W

    # Normalize for display
    disp_norm = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
    depth_u8  = np.uint8(disp_norm)

    # Show
    show_image(depth_u8, f"Depth {tag}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"depth_{tag}.png"
    cv.imwrite(str(out_path), depth_u8)
    print(f"[+] Saved depth map to {out_path}")

    return depth_map

def compute_reprojection_error(disp: np.ndarray,
                               Q: np.ndarray,
                               R: np.ndarray,
                               t: np.ndarray,
                               K: np.ndarray,
                               valid_mask: np.ndarray):
    """
    disp:      H×W float disparity map
    Q:         4×4 reprojection matrix returned by stereoRectify
    R, t:      3×3 and 3×1 extrinsic used for projection.
              (For rectified consistency checks, you typically pass R=I, t=0.)
    K:         3×3 intrinsic used for projection.
    valid_mask: H×W bool mask of which pixels were used to build your point cloud

    Returns
    -------
    errors:    (N,) reprojection errors in pixels
    """
    # 1) reproject disparity → 3D in camera coords
    pts3D = cv.reprojectImageTo3D(disp, Q)  # H×W×3
    H,W = disp.shape

    # 2) pick only the valid ones
    valid = valid_mask
    xyz = pts3D[valid]                      # N×3
    # also build the corresponding original pixel locations (u,v)
    ys, xs = np.nonzero(valid)              # each is (N,)
    uv_orig = np.stack([xs, ys], axis=1).astype(np.float32)  # N×2

    # 3) project back into the image:
    # OpenCV wants rvec (Rodrigues) + tvec
    rvec, _ = cv.Rodrigues(R)
    imgpts, _ = cv.projectPoints(xyz, rvec, t, K, distCoeffs=None)
    uv_proj = imgpts.reshape(-1,2)          # N×2

    # 4) compute per‐point reprojection error
    errs = np.linalg.norm(uv_proj - uv_orig, axis=1)  # N
    return errs

def fuse_pointclouds(all_pcs: list[o3d.geometry.PointCloud],
                     voxel_size: float = 0.02,
                     nb_neighbors: int = 16,
                     std_ratio: float = 2.0,
                     out_path: Path = Path("data/output")/"fused.ply",
                     downsample = True):
    """
    all_pcs: list of o3d.geometry.PointCloud objects
    voxel_size: size in meters for voxel down‐sampling
    nb_neighbors, std_ratio: for statistical outlier removal
    """
    # 1) concatenate into one big cloud
    fused = o3d.geometry.PointCloud()
    for pc in all_pcs:
        fused += pc
    
    print(f"DEBUG: Bounding‐box diagonal = voxel_size → {voxel_size:.6f}")
    
    if downsample and voxel_size > 0:
        fused = fused.voxel_down_sample(voxel_size)

    # 3) optionally remove statistical outliers
    fused, ind = fused.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    # 4) (optional) re‐estimate normals if you plan to mesh
    fused.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30)
    )

    # 5) save the result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), fused)
    print(f"[+] Fused point cloud saved to {out_path}")

    return fused

def compute_and_save_pointcloud(disp: np.ndarray,
                                Q: np.ndarray,
                                R: np.ndarray,
                                t: np.ndarray,
                                K: np.ndarray,
                                color_img: np.ndarray,
                                out_dir: Path,
                                tag: str,
                                rect_R: Optional[np.ndarray] = None,
                                K_rect: Optional[np.ndarray] = None,
                max_depth: float = 5.0,
                to_world: bool = True,
                show_reprojection_debug: bool = False):
    """
    disp: float32 disparity map (H×W)
    Q:    4×4 reprojection matrix
    R, t: COLMAP world-to-camera pose for the *left* image (x_cam = R x_world + t)
    color_img: the rectified color image (H×W×3 BGR)
        rect_R: rectification rotation (R1 returned by cv.stereoRectify)
            that maps original left-camera coords -> rectified left-camera coords.
            If provided, points are rotated back via rect_R.T.
        K_rect: rectified intrinsic matrix (typically P1[:3,:3]) used for optional
            reprojection-error debug in rectified pixel coordinates.
    """
    # Reproject to 3D
    pts3D = cv.reprojectImageTo3D(disp, Q)  # H×W×3
    # mask  = disp > disp.min()               # valid disparity

    Z = pts3D[..., 2]

    #  1) positive disparity
    m1 = disp > 0
    #  2) finite depth
    m2 = np.isfinite(Z)
    # (optional) clamp maximum depth if you really need it:
    # m3 = Z < max_depth

    # final valid mask: drop the Z<max_depth test if it's filtering everything
    valid = m1 & m2  # & m3  if you still want the clamp

    if show_reprojection_debug and K_rect is not None:
        errs = compute_reprojection_error(
            disp,
            Q,
            np.eye(3, dtype=float),
            np.zeros((3, 1), dtype=float),
            K_rect,
            valid,
        )
        print(f"Reprojection (rectified): mean = {errs.mean():.2f}px,  max = {errs.max():.2f}px")
        import matplotlib.pyplot as plt
        plt.hist(errs, bins=50)
        plt.title("Reprojection error distribution (rectified)")
        plt.xlabel("pixels")
        plt.ylabel("count")
        plt.show()

    xyz_rect = pts3D[valid]  # N×3 in rectified left-camera coordinates

    # Rotate rectified coords back to the original left-camera coords if rect_R is provided.
    # stereoRectify returns rect_R such that: x_rect = rect_R * x_cam
    if rect_R is not None:
        xyz_cam = (rect_R.T @ xyz_rect.T).T
    else:
        xyz_cam = xyz_rect

    # Transform from left-camera coords to COLMAP world coords.
    # COLMAP pose convention: x_cam = R * x_world + t  =>  x_world = R^T * (x_cam - t)
    if to_world:
        xyz = (R.T @ (xyz_cam.T - t.reshape(3, 1))).T
    else:
        xyz = xyz_cam

    rgb   = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)[valid]  # N×3

    # Build Open3D point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"cloud_{tag}.ply"
    o3d.io.write_point_cloud(str(ply_path), pc)
    print(f"[+] Saved point cloud to {ply_path}")

    return pc

def compute_show_save_disparity(r1, r2, out_dir: Path, tag: str,
                                min_disp=0, num_disp=128, block_size=5):
    """
    Given two rectified BGR images r1, r2:
      - compute disparity with StereoSGBM
      - normalize & show via show_image()
      - save a PNG under out_dir/disp_{tag}.png
    """
    # 1) convert to gray
    if r1.ndim == 3 and r1.shape[2] == 3:
        g1 = cv.cvtColor(r1, cv.COLOR_BGR2GRAY)
    else:
        g1 = r1.copy()
    if r2.ndim == 3 and r2.shape[2] == 3:
        g2 = cv.cvtColor(r2, cv.COLOR_BGR2GRAY)
    else:
        g2 = r2.copy()

    # 2) create matcher
    stereo = cv.StereoSGBM_create(
        minDisparity    = min_disp,
        numDisparities  = num_disp,  # must be divisible by 16
        blockSize       = block_size,
        P1              = 8 * block_size**2,
        P2              = 32 * block_size**2,
        disp12MaxDiff   = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange      = 32,
        preFilterCap      = 63,
        mode              = cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 3) compute & convert to float32
    disp16 = stereo.compute(g1, g2)           # int16
    disp   = disp16.astype(np.float32) / 16.0

    # 4) normalize to 0–255 for visualization
    disp_norm = cv.normalize(disp, None, alpha=0, beta=255,
                             norm_type=cv.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_norm)

    # 5) show it
    show_image(disp_uint8, f"Disparity {tag}")

    # 6) save it
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"disp_{tag}.png"
    cv.imwrite(str(out_path), disp_uint8)
    print(f"[+] Saved disparity map to {out_path}")

    # return raw disp if you need it for 3D later
    return disp

def show_image(img, win_name, max_dim=1200):
    """Utility to show a possibly large image in a resizable window, down-sampled."""
    h, w = img.shape[:2]
    scale = min(max_dim / max(w, h), 1.0)
    disp = cv.resize(img, (int(w*scale), int(h*scale)), interpolation=cv.INTER_AREA)
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, disp)
    cv.waitKey(0)
    cv.destroyWindow(win_name)

def draw_unrectified_epilines(img1, img2, F, num=10):
    """Draws epipolar lines on img1/img2 given fundamental matrix F."""
    h, w = img1.shape[:2]
    # sample random points in img1
    pts1 = np.column_stack((np.random.randint(0, w, num),
                             np.random.randint(0, h, num))).astype(np.float32)
    # lines in img2 for these pts
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F).reshape(-1,3)
    img1_pts = img1.copy()
    img2_lines = img2.copy()
    for p in pts1:
        cv.circle(img1_pts, tuple(p.astype(int)), 5, (0,255,0), -1)
    for a,b,c in lines2:
        # line: a x + b y + c = 0
        y0 = int(-c/b)
        y1 = int(-(a*w + c)/b)
        cv.line(img2_lines, (0,y0),(w,y1), (0,255,0), 3, lineType=cv.LINE_AA)

    # sample random points in img2
    pts2 = np.column_stack((np.random.randint(0, w, num),
                             np.random.randint(0, h, num))).astype(np.float32)
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F).reshape(-1,3)
    img2_pts = img2.copy()
    img1_lines = img1.copy()
    for p in pts2:
        cv.circle(img2_pts, tuple(p.astype(int)), 5, (0,0,255), -1)
    for a,b,c in lines1:
        y0 = int(-c/b)
        y1 = int(-(a*w + c)/b)
        cv.line(img1_lines, (0,y0),(w,y1), (0,0,255), 3, lineType=cv.LINE_AA)

    # stack for display
    left = np.hstack((img1_pts, img1_lines))
    right= np.hstack((img2_pts, img2_lines))
    return np.vstack((left, right))

def draw_rectified_epilines(r1, r2, num=10):
    """Draws horizontal epipolar lines on rectified images."""
    h, w = r1.shape[:2]
    img1 = r1.copy(); img2 = r2.copy()
    # pick y's equally spaced
    ys = np.linspace(0, h-1, num, dtype=int)
    for y in ys:
        cv.line(img1, (0,y),(w,y), (0,0,255), 3, lineType=cv.LINE_AA)
        cv.line(img2, (0,y),(w,y), (0,0,255), 3, lineType=cv.LINE_AA)
    return np.hstack((img1, img2))

def rectify_and_visualize(data_dir, cams, poses, present, all_pcs, pairs=3):
    valid = [p for p in poses if Path(p['name']).name in present]
    valid.sort(key=lambda p: Path(p['name']).name)

    for i in range(min(pairs, len(valid)-1)):
        p1, p2 = valid[i], valid[i+1]
        n1, n2 = Path(p1['name']).name, Path(p2['name']).name
        cam1, cam2 = cams[p1['cam_id']], cams[p2['cam_id']]

        # load images (assumed undistorted)
        img1 = cv.imread(str(data_dir/"images"/n1))
        img2 = cv.imread(str(data_dir/"images"/n2))

        K1, d1 = cam1['K'], cam1['dist']
        K2, d2 = cam2['K'], cam2['dist']
        R1, t1 = p1['R'], p1['t']
        R2, t2 = p2['R'], p2['t']

        # compute relative pose
        R_rel = R2 @ R1.T
        T_rel = t2 - R_rel @ t1

        # fundamental matrix F = K2^{-T} [T]_x R K1^{-1}
        t = T_rel.ravel()
        tx = np.array([[   0, -t[2],  t[1]],
                       [ t[2],    0, -t[0]],
                       [-t[1],  t[0],    0]])
        E = tx @ R_rel
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

        # # 1) show epipolar lines (unrectified)
        # epi_unrect = draw_unrectified_epilines(img1, img2, F, num=8)
        # show_image(epi_unrect, f"Epipolar Lines (Unrect) {n1}|{n2}")

        g1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(g1, maxCorners=50, qualityLevel=0.01, minDistance=20)
        corners = corners.reshape(-1,2)
        pts = corners[random.sample(range(len(corners)), 10)]

        # 2) draw unrectified epilines for those pts
        epiL, epiR = draw_epilines_for_keypoints(img1, img2, F, pts)
        show_image(np.hstack((epiL, epiR)), f"Unrectified Epilines")

        # 2) rectify stereo pair
        h, w = img1.shape[:2]
        R1n, R2n, P1, P2, Q, _, _ = cv.stereoRectify(
            K1, d1, K2, d2, (w, h), R_rel, T_rel,
            flags=cv.CALIB_ZERO_DISPARITY, alpha=0)
        map1x, map1y = cv.initUndistortRectifyMap(
            K1, d1, R1n, P1, (w, h), cv.CV_32FC1)
        map2x, map2y = cv.initUndistortRectifyMap(
            K2, d2, R2n, P2, (w, h), cv.CV_32FC1)
        r1 = cv.remap(img1, map1x, map1y, cv.INTER_LINEAR)
        r2 = cv.remap(img2, map2x, map2y, cv.INTER_LINEAR)

        # BEFORE showing them, save the full-res rectified images:
        output_rect = data_dir / "output" / "rectified"
        output_rect.mkdir(parents=True, exist_ok=True)

        out1 = output_rect / f"rect_{n1}"
        out2 = output_rect / f"rect_{n2}"
        cv.imwrite(str(out1), r1)
        cv.imwrite(str(out2), r2)
        print(f"[+] Saved rectified images:\n    {out1}\n    {out2}")

        # # warp your original keypoints pts (N×2) into rectified coords:
        # # use cv.remap on both u‐ and v‐maps
        # u_map = map1x
        # v_map = map1y
        # # interpolate each point
        # pts_rect = []
        # for (u,v) in pts:
        #     # note: remap wants a small image, so we build a tiny 1×1 image holding u or v
        #     u_warp = cv.remap(np.array([[u]],dtype=np.float32), u_map, v_map, cv.INTER_LINEAR)[0,0]
        #     v_warp = cv.remap(np.array([[v]],dtype=np.float32), u_map, v_map, cv.INTER_LINEAR)[0,0]
        #     pts_rect.append( (u_warp, v_warp) )
        # pts_rect = np.array(pts_rect, dtype=np.float32)

        # # debug‐print them:
        # print("[DBG] pts_rect pre‐clamp:\n", pts_rect)

        # h_img, w_img = r1.shape[:2]
        # # clamp to valid integer pixel rows
        # ys = pts_rect[:,1]
        # ys = np.nan_to_num(ys, nan=0.0)
        # ys = np.clip(ys, 0, h_img-1).astype(int)

        # print("[DBG] clamped ys:", ys)

        # # now draw only those valid horizontal lines:
        # outL = r1.copy()
        # outR = r2.copy()
        # th = 3
        # for y in ys:
        #     cv.line(outL, (0,y),(w_img,y),(0,0,255),th)
        #     cv.line(outR, (0,y),(w_img,y),(0,0,255),th)

        # show_image(np.hstack((outL,outR)), "Rectified Horizontal Epilines")

        # 3) show rectified pair
        # show_image(np.hstack((r1,r2)), f"Rectified Pair {n1}|{n2}")

        # 4) show horizontal epipolar lines
        epi_rect = draw_rectified_epilines(r1, r2, num=8)
        show_image(epi_rect, f"Epipolar Lines (Rect) {n1}|{n2}")

        # 5) compute, show, and save disparity

        r1g = cv.cvtColor(r1, cv.COLOR_BGR2GRAY)
        r2g = cv.cvtColor(r2, cv.COLOR_BGR2GRAY)
        
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        r1g = clahe.apply(r1g)
        r2g = clahe.apply(r2g)

        r1g = cv.fastNlMeansDenoising(r1g, None, h=5)
        r2g = cv.fastNlMeansDenoising(r2g, None, h=5)

        disp = compute_show_save_disparity(
            r1g, r2g,
            out_dir = data_dir/"output"/"disparities",
            tag     = f"{n1[:-4]}_{n2[:-4]}",
            min_disp   = 0,
            num_disp   = 128,
            block_size = 5
        )

        # 6) compute, show & save depth map
        # depth = compute_show_save_depth(
        #     disp, Q,
        #     out_dir = data_dir/"output"/"depthmaps",
        #     tag     = f"{n1[:-4]}_{n2[:-4]}"
        # )

        # 7) compute & save 3D point cloud
        pc = compute_and_save_pointcloud(
            disp, Q, R1, t1, K1,
            r1,
            rect_R=R1n,
            K_rect=P1[:3, :3],
            out_dir = data_dir/"output"/"pointclouds",
            tag     = f"{n1[:-4]}_{n2[:-4]}",
            to_world=True,
            show_reprojection_debug=False,
        )
        # npts = np.asarray(pc.points).shape[0]
        # print(f"   → In‐memory PointCloud for {n1}-{n2} has {npts} points")
        all_pcs.append(pc)

    # print(">>> Total point clouds collected:", len(all_pcs))
    # for idx, pc in enumerate(all_pcs):
    #     print(f"     cloud {idx}: {np.asarray(pc.points).shape[0]} pts")

if __name__ == "__main__":
    data = Path("data")
    cams   = load_cameras(data / "cameras.txt")
    poses  = load_image_poses(data / "images.txt")

    # Build the set of files actually on disk
    img_folder = data / "images"
    present = {p.name for p in img_folder.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    # 1) Intrinsics Parameters
    print("=== Camera Intrinsic Matrices ===")
    for cam_id, cam in cams.items():
        print(f"\nCamera ID {cam_id}:")
        print(cam['K'])

    # 2) Extrinsic Parameters for present images only
    print("\n=== Extrinsic Homogeneous Matrices for present images ===")
    for p in poses:
        img_name = Path(p['name']).name        # <-- take basename only
        if img_name not in present:
            continue

        R = p['R']           # 3×3
        t = p['t'].ravel()   # (3,)
        H = np.eye(4)
        H[:3, :3] = R
        H[:3,  3] = t
        print(f"\nImage: {img_name}")
        print(H)

    # finally visualize epipolar before/after rectification

    all_pcs = []
    rectify_and_visualize(data, cams, poses, present, all_pcs, pairs=25)

    # all_pcs = [pc for pc in all_pcs if np.asarray(pc.points).shape[0] > 0]
    # if not all_pcs:
    #     raise RuntimeError("No non-empty point clouds to fuse! Check your masking.")

    fused_pc = fuse_pointclouds(
        all_pcs,
        voxel_size=0.02,      # try 2 cm for ETH3D office
        nb_neighbors=16,
        std_ratio=2.0,
        out_path=Path("data")/"output"/"fused.ply"
    )
