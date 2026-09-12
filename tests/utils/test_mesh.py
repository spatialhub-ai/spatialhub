"""Comprehensive unit tests for spatialhub.utils.mesh module."""

import numpy as np
import pytest

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh dependency required for mesh tests")

if TRIMESH_AVAILABLE:
    from spatialhub.utils.mesh import (
        MeshArrays,
        center_mesh,
        compute_mesh_diameter,
        compute_obb,
        invert_transform,
        load_mesh,
        look_at,
        prepare_mesh_arrays,
        read_mesh,
        sample_sphere_poses,
        scale_mesh,
        to_single_mesh,
    )


class TestMeshAtomicOperations:
    """Test suite for atomic mesh reading, scaling, and centering."""

    @pytest.fixture
    def sample_box(self) -> "trimesh.Trimesh":
        box = trimesh.creation.box(extents=[2.0, 4.0, 6.0])
        box.apply_translation([10.0, 20.0, 30.0])
        return box

    def test_read_mesh_and_to_single_mesh(self, sample_box: "trimesh.Trimesh"):
        mesh = read_mesh(sample_box)
        assert len(mesh.vertices) == len(sample_box.vertices)
        assert len(mesh.faces) == len(sample_box.faces)

    def test_to_single_mesh_scene(self, sample_box: "trimesh.Trimesh"):
        scene = trimesh.Scene(geometry=[sample_box])
        single_mesh = to_single_mesh(scene)
        assert isinstance(single_mesh, trimesh.Trimesh)
        assert len(single_mesh.vertices) == len(sample_box.vertices)

    def test_to_single_mesh_empty_scene_raises(self):
        empty_scene = trimesh.Scene()
        with pytest.raises(ValueError, match="loaded trimesh Scene is empty"):
            to_single_mesh(empty_scene)

    def test_to_single_mesh_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected trimesh.Trimesh or trimesh.Scene"):
            to_single_mesh("invalid_object")

    @pytest.mark.parametrize(
        "unit, expected_scale",
        [
            ("m", 1.0),
            ("M", 1.0),
            ("cm", 0.01),
            ("CM", 0.01),
            ("mm", 0.001),
            ("MM", 0.001),
            (0.5, 0.5),
        ],
    )
    def test_scale_mesh_units(self, sample_box: "trimesh.Trimesh", unit: str | float, expected_scale: float):
        scaled = scale_mesh(sample_box, model_unit=unit)
        expected_extents = sample_box.extents * expected_scale
        assert np.allclose(scaled.extents, expected_extents, atol=1e-5)

    def test_scale_mesh_invalid_unit_raises(self, sample_box: "trimesh.Trimesh"):
        with pytest.raises(TypeError, match="model_unit must be a string identifier"):
            scale_mesh(sample_box, model_unit=["invalid"])

    def test_center_mesh(self, sample_box: "trimesh.Trimesh"):
        centered, offset = center_mesh(sample_box)
        assert np.allclose(centered.bounding_box.centroid, [0.0, 0.0, 0.0], atol=1e-5)
        assert np.allclose(offset, -sample_box.bounding_box.centroid, atol=1e-5)

    def test_compute_mesh_diameter(self, sample_box: "trimesh.Trimesh"):
        # Box extents: [2, 4, 6] -> diagonal diameter = sqrt(2^2 + 4^2 + 6^2) = sqrt(56) ~ 7.4833
        diameter = compute_mesh_diameter(sample_box)
        expected_diameter = np.sqrt(2.0**2 + 4.0**2 + 6.0**2)
        assert np.isclose(diameter, expected_diameter, atol=1e-3)

    def test_load_mesh_composite(self, sample_box: "trimesh.Trimesh"):
        loaded = load_mesh(sample_box, model_unit="mm", center=True)
        assert np.allclose(loaded.bounding_box.centroid, [0.0, 0.0, 0.0], atol=1e-5)
        assert np.isclose(loaded.extents[0], 0.002)


class TestOrientedBoundingBoxAndTransforms:
    """Test suite for OBB computation, camera look-at, and transformation matrix math."""

    @pytest.fixture
    def canonical_box(self) -> "trimesh.Trimesh":
        return trimesh.creation.box(extents=[1.0, 2.0, 3.0])

    def test_compute_obb(self, canonical_box: "trimesh.Trimesh"):
        corners, extents, to_origin = compute_obb(canonical_box)
        assert corners.shape == (8, 3)
        assert extents.shape == (3,)
        assert to_origin.shape == (4, 4)
        assert np.allclose(sorted(extents), [1.0, 2.0, 3.0], atol=1e-3)

    def test_look_at_standard_and_degenerate(self):
        # Camera at (0, 0, 5) looking at (0, 0, 0)
        c2w = look_at(cam_location=np.array([0.0, 0.0, 5.0]), target_point=np.array([0.0, 0.0, 0.0]))
        assert c2w.shape == (4, 4)
        assert np.allclose(c2w[:3, 3], [0.0, 0.0, 5.0])

        # Degenerate collinear case: cam_location == target_point
        c2w_collinear = look_at(cam_location=np.array([0.0, 0.0, 0.0]), target_point=np.array([0.0, 0.0, 0.0]))
        assert c2w_collinear.shape == (4, 4)

    def test_invert_transform_single(self):
        t_mat = np.eye(4, dtype=np.float32)
        t_mat[:3, 3] = [1.0, 2.0, 3.0]
        # 90 deg rotation around Z
        t_mat[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

        t_inv = invert_transform(t_mat)
        identity_test = t_mat @ t_inv
        assert np.allclose(identity_test, np.eye(4), atol=1e-5)

    def test_invert_transform_batch(self):
        t_batch = np.tile(np.eye(4, dtype=np.float32)[None], (3, 1, 1))
        t_batch[:, :3, 3] = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

        inv_batch = invert_transform(t_batch)
        assert inv_batch.shape == (3, 4, 4)
        for i in range(3):
            assert np.allclose(t_batch[i] @ inv_batch[i], np.eye(4), atol=1e-5)

    def test_invert_transform_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Expected transformation matrix"):
            invert_transform(np.zeros((3, 3), dtype=np.float32))


class TestSpherePoseSampling:
    """Test suite for sample_sphere_poses."""

    def test_sample_sphere_poses_fibonacci(self):
        poses_obj = sample_sphere_poses(num_viewpoints=10, radius=1.0, sampling_method="fibonacci", pose_type="object_pose")
        assert poses_obj.shape == (10, 4, 4)
        assert poses_obj.dtype == np.float32

        poses_cam = sample_sphere_poses(num_viewpoints=10, radius=1.0, sampling_method="fibonacci", pose_type="camera_pose")
        assert poses_cam.shape == (10, 4, 4)

    def test_sample_sphere_poses_icosphere(self):
        poses_ico = sample_sphere_poses(num_viewpoints=12, radius=0.8, sampling_method="icosphere", pose_type="camera_pose")
        assert poses_ico.ndim == 3 and poses_ico.shape[1:] == (4, 4)

    def test_sample_sphere_poses_invalid_num_viewpoints_raises(self):
        with pytest.raises(ValueError, match="num_viewpoints must be greater than 1"):
            sample_sphere_poses(num_viewpoints=1, sampling_method="fibonacci")

    def test_sample_sphere_poses_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported sampling_method"):
            sample_sphere_poses(num_viewpoints=10, sampling_method="invalid_method")

    def test_sample_sphere_poses_invalid_pose_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported pose_type"):
            sample_sphere_poses(num_viewpoints=10, pose_type="invalid_type")


class TestPrepareMeshArrays:
    """Test suite for prepare_mesh_arrays and MeshArrays container."""

    def test_prepare_mesh_arrays_vertex_colors(self):
        box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        mesh_arr = prepare_mesh_arrays(box)

        assert isinstance(mesh_arr, MeshArrays)
        assert mesh_arr.pos.shape == (len(box.vertices), 3)
        assert mesh_arr.faces.shape == (len(box.faces), 3)
        assert mesh_arr.vnormals.shape == (len(box.vertices), 3)
        assert mesh_arr.vertex_color is not None
        assert mesh_arr.vertex_color.shape == (len(box.vertices), 3)

        buf_dict = mesh_arr.as_dict()
        assert "pos" in buf_dict
        assert "faces" in buf_dict
        assert "vnormals" in buf_dict
