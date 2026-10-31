# Geometry

The `spatialhub.utils.geometry` module provides geometric functions for camera pose estimation from point correspondences.

## Functions

### `estimate_relative_pose_2d`

Estimates the relative camera pose between two calibrated views from 2D point correspondences.

```python
def estimate_relative_pose_2d(
    points_a: np.ndarray,
    points_b: np.ndarray,
    intrinsics_a: np.ndarray,
    intrinsics_b: np.ndarray | None = None,
    threshold_px: float = 1.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray]
```

#### Mathematical Formulation

Given pixel coordinates $\mathbf{x} = [u, v, 1]^T$ and camera intrinsic matrices $K_A, K_B$, coordinates are normalized to the canonical camera plane $z = 1$:

$$\tilde{\mathbf{x}}_A = K_A^{-1} \mathbf{x}_A, \quad \tilde{\mathbf{x}}_B = K_B^{-1} \mathbf{x}_B$$

Corresponding normalized rays satisfy the epipolar constraint:

$$\tilde{\mathbf{x}}_B^T E \, \tilde{\mathbf{x}}_A = 0$$

where $E = [t]_\times R$ is the Essential matrix relating the two calibrated views.

The Essential matrix is estimated using 5-point RANSAC on normalized coordinates, with reprojection threshold scaled by the mean focal length. When minimal point samples yield multiple algebraic roots, each candidate solution is evaluated against the cheirality constraint, ensuring triangulated points have positive scene depth in both views.

The resulting transformation matrix $T \in \mathrm{SE}(3)$ maps 3D points from camera frame A to camera frame B:

$$P_B = T P_A = \begin{bmatrix} R & t \\ \mathbf{0}_{1 \times 3} & 1 \end{bmatrix} P_A$$

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `points_a` | `(N, 2)` | `float32` or `float64` | 2D pixel coordinates in image A. |
| `points_b` | `(N, 2)` | `float32` or `float64` | 2D pixel coordinates in image B. |
| `intrinsics_a` | `(3, 3)` | `float32` or `float64` | Intrinsic calibration matrix $K_A$ for camera A. |
| `intrinsics_b` | `(3, 3)` or `None` | `float32` or `float64` | Intrinsic calibration matrix $K_B$ for camera B. Defaults to `intrinsics_a`. |
| `threshold_px` | `float` | - | Reprojection error threshold in pixels for RANSAC inliers. |
| `confidence` | `float` | - | RANSAC confidence level (probability of selecting an outlier-free sample). |

#### Outputs

Tuple containing:

- `transform`: `(4, 4)` array, dtype `float32`. Transformation matrix $T \in \mathrm{SE}(3)$ mapping points from camera frame A to camera frame B, with unit translation direction ($\|t\| = 1$).
- `inlier_mask`: `(N,)` boolean array. `True` for correspondences satisfying epipolar geometry and positive scene depth.

#### Example

```python
import numpy as np
from spatialhub.utils import estimate_relative_pose_2d, project_points_to_image

# camera intrinsics
intrinsics = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

# synthetic 3D scene points
rng = np.random.default_rng(seed=42)
pts_3d = rng.uniform(-0.5, 0.5, size=(50, 3)).astype(np.float32)
pts_3d[:, 2] += 2.5

# project to view A
points_a, _ = project_points_to_image(pts_3d, intrinsics)

# ground truth relative motion (rotation and translation)
transform_gt = np.eye(4, dtype=np.float32)
transform_gt[:3, :3] = [
    [0.9961947, 0.0, 0.0871557],
    [0.0, 1.0, 0.0],
    [-0.0871557, 0.0, 0.9961947],
]
transform_gt[:3, 3] = [0.2, 0.0, 0.1]

# transform and project to view B
points_b, _ = project_points_to_image(pts_3d, intrinsics, pose=transform_gt)

# estimate relative pose
transform, inlier_mask = estimate_relative_pose_2d(
    points_a=points_a,
    points_b=points_b,
    intrinsics_a=intrinsics,
    threshold_px=1.0,
)

print("Inliers:", np.count_nonzero(inlier_mask), "/", len(points_a))
print("Rotation Matrix:\n", transform[:3, :3])
print("Unit Translation Vector:", transform[:3, 3])
```

---

### `solve_pnp_ransac`

Estimates camera pose from 3D points and their corresponding 2D image projections using Perspective-n-Point (PnP) with RANSAC.

```python
def solve_pnp_ransac(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    reprojection_error_px: float = 3.0,
    confidence: float = 0.999,
    max_iterations: int = 100,
    flags: int = cv2.SOLVEPNP_EPNP,
    refine_lm: bool = True,
) -> tuple[np.ndarray, np.ndarray]
```

#### Mathematical Formulation

Given 3D points $P_i \in \mathbb{R}^3$ in the source reference frame and observed 2D pixel coordinates $p_i \in \mathbb{R}^2$, the solver estimates rotation $R \in \mathrm{SO}(3)$ and translation $t \in \mathbb{R}^3$ by minimizing reprojection error over the inlier set $\mathcal{I}$:

$$\min_{R, t} \sum_{i \in \mathcal{I}} \left\| p_i - \pi(K, R P_i + t, D) \right\|^2$$

where $\pi$ projects a 3D point using camera intrinsics $K$ and distortion coefficients $D$.

RANSAC samples minimal subsets using the solver specified by `flags` (defaulting to Efficient PnP). When `refine_lm=True`, the pose is refined on all inliers using Levenberg-Marquardt optimization.

The resulting transformation matrix $T \in \mathrm{SE}(3)$ maps 3D points from the source reference frame to the camera frame:

$$P_{\text{cam}} = T P_{\text{obj}} = \begin{bmatrix} R & t \\ \mathbf{0}_{1 \times 3} & 1 \end{bmatrix} P_{\text{obj}}$$

#### Inputs

| Argument | Type / Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `points_3d` | `(N, 3)` | `float32` or `float64` | 3D points in the source reference frame. |
| `points_2d` | `(N, 2)` | `float32` or `float64` | Corresponding 2D pixel coordinates in the camera frame. |
| `intrinsics` | `(3, 3)` | `float32` or `float64` | Camera intrinsic calibration matrix $K$. |
| `dist_coeffs` | `(4,)`, `(5,)`, `(8,)` or `None` | `float32` or `float64` | Lens distortion coefficients $(k_1, k_2, p_1, p_2[, k_3, \dots])$. Set to `None` for undistorted points. |
| `reprojection_error_px` | `float` | - | Maximum reprojection error in pixels to consider a point an inlier. |
| `confidence` | `float` | - | RANSAC confidence level (probability of finding an outlier-free sample). |
| `max_iterations` | `int` | - | Maximum number of RANSAC iterations. |
| `flags` | `int` | - | PnP solver method (e.g. `cv2.SOLVEPNP_EPNP`, `cv2.SOLVEPNP_ITERATIVE`). |
| `refine_lm` | `bool` | - | Whether to refine the pose estimate on inliers using Levenberg-Marquardt optimization. |

#### Outputs

Tuple containing:

- `transform`: `(4, 4)` array, dtype `float32`. Transformation matrix $T \in \mathrm{SE}(3)$ mapping points from the source reference frame to the camera frame, with metric translation.
- `inlier_mask`: `(N,)` boolean array. `True` for correspondences classified as inliers.

#### Example

```python
import numpy as np
from spatialhub.utils import project_points_to_image, solve_pnp_ransac

# camera intrinsics
intrinsics = np.array([
    [550.0, 0.0, 320.0],
    [0.0, 550.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

# 3D points in the object frame
rng = np.random.default_rng(seed=100)
points_3d = rng.uniform(-0.8, 0.8, size=(40, 3)).astype(np.float32)
points_3d[:, 2] += 2.0

# ground truth transformation matrix mapping object to camera
transform_gt = np.array([
    [0.9984795, -0.0199987,  0.0513972,  0.15],
    [0.0209971,  0.9995594, -0.0209971, -0.10],
    [-0.0509986, 0.0219468,  0.9984577,  0.35],
    [0.0,        0.0,        0.0,        1.0 ],
], dtype=np.float32)

# project 3D points to 2D image coordinates
points_2d, _ = project_points_to_image(points_3d, intrinsics, pose=transform_gt)

# estimate camera pose with PnP RANSAC
transform, inlier_mask = solve_pnp_ransac(
    points_3d=points_3d,
    points_2d=points_2d,
    intrinsics=intrinsics,
    reprojection_error_px=2.0,
)

print("Inliers:", np.count_nonzero(inlier_mask), "/", len(points_3d))
print("Estimated Transformation Matrix:\n", transform)
```
