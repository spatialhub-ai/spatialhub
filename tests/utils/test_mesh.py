"""Comprehensive unit tests for spatialhub.utils.mesh module."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    pytest.importorskip("trimesh", reason="trimesh dependency required for mesh tests") is None,
    reason="trimesh dependency required",
)

import trimesh

from spatialhub.utils.mesh import (
    MeshBuffers,
    OrientedBoundingBox,
    center_mesh,
    compute_mesh_diameter,
    compute_obb,
    load_mesh,
    prepare_mesh_buffers,
    scale_mesh,
)


@pytest.fixture
def sample_box_mesh() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[2.0, 4.0, 6.0])
    box.apply_translation([10.0, 20.0, 30.0])
    return box


@pytest.fixture
def temp_obj_file(tmp_path: Path, sample_box_mesh: trimesh.Trimesh) -> Path:
    obj_path = tmp_path / "box.obj"
    sample_box_mesh.export(str(obj_path))
    return obj_path


class TestMeshContainers:
    """Test suite for MeshBuffers and OrientedBoundingBox containers."""

    def test_mesh_buffers_as_dict(self) -> None:
        v = np.zeros((4, 3), dtype=np.float32)
        f = np.zeros((2, 3), dtype=np.int32)
        n = np.zeros((4, 3), dtype=np.float32)
        buf = MeshBuffers(vertices=v, faces=f, normals=n)

        d = buf.as_dict()
        assert "vertices" in d
        assert "faces" in d
        assert "normals" in d
        assert "texture" not in d
        assert "uvs" not in d

    def test_oriented_bounding_box_unpacking(self) -> None:
        corners = np.zeros((8, 3), dtype=np.float32)
        extents = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        to_origin = np.eye(4, dtype=np.float32)

        obb = OrientedBoundingBox(corners=corners, extents=extents, to_origin=to_origin)
        c, e, t = obb
        assert np.array_equal(c, corners)
        assert np.array_equal(e, extents)
        assert np.array_equal(t, to_origin)

        d = obb.as_dict()
        assert len(d) == 3


class TestLoadMesh:
    """Test suite for load_mesh."""

    def test_load_existing_mesh(self, temp_obj_file: Path) -> None:
        mesh = load_mesh(temp_obj_file)
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Mesh file not found"):
            load_mesh(tmp_path / "nonexistent.obj")

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="path must be a str or Path"):
            load_mesh(12345)  # type: ignore[arg-type]


class TestScaleMesh:
    """Test suite for scale_mesh."""

    def test_scale_units(self) -> None:
        # mm -> m converts factor 0.001
        box = trimesh.creation.box(extents=[1000.0, 1000.0, 1000.0])
        scaled = scale_mesh(box, model_unit="mm")
        assert np.allclose(scaled.extents, [1.0, 1.0, 1.0])

        # cm -> m converts factor 0.01
        box_cm = trimesh.creation.box(extents=[100.0, 100.0, 100.0])
        scaled_cm = scale_mesh(box_cm, model_unit="cm")
        assert np.allclose(scaled_cm.extents, [1.0, 1.0, 1.0])

        # m -> m converts factor 1.0
        box_m = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        scaled_m = scale_mesh(box_m, model_unit="m")
        assert np.allclose(scaled_m.extents, [1.0, 1.0, 1.0])

    def test_custom_float_multiplier(self) -> None:
        box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        scaled = scale_mesh(box, model_unit=0.5)
        assert np.allclose(scaled.extents, [1.0, 1.0, 1.0])

    def test_invalid_unit_raises(self) -> None:
        box = trimesh.creation.box()
        with pytest.raises(ValueError, match="Unsupported model_unit 'inches'"):
            scale_mesh(box, model_unit="inches")

        with pytest.raises(TypeError, match="model_unit must be a string identifier"):
            scale_mesh(box, model_unit=None)  # type: ignore[arg-type]


class TestCenterMesh:
    """Test suite for center_mesh."""

    def test_center_mesh_moves_centroid_to_origin(self, sample_box_mesh: trimesh.Trimesh) -> None:
        initial_centroid = sample_box_mesh.bounding_box.centroid.copy()
        assert not np.allclose(initial_centroid, [0.0, 0.0, 0.0])

        centered, offset = center_mesh(sample_box_mesh)
        assert np.allclose(centered.bounding_box.centroid, [0.0, 0.0, 0.0], atol=1e-5)
        assert np.allclose(offset, -initial_centroid, atol=1e-5)


class TestComputeMeshDiameter:
    """Test suite for compute_mesh_diameter."""

    def test_cube_bounding_diameter(self) -> None:
        # Cube of 2x2x2 has diameter sqrt(2^2 + 2^2 + 2^2) = sqrt(12) ~= 3.4641
        cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        diameter = compute_mesh_diameter(cube, num_dirs=128)
        expected = np.sqrt(12.0)
        assert np.isclose(diameter, expected, atol=0.1)

    def test_empty_mesh_raises(self) -> None:
        empty = trimesh.Trimesh()
        with pytest.raises(ValueError, match="Cannot compute diameter of an empty mesh"):
            compute_mesh_diameter(empty)


class TestComputeOBB:
    """Test suite for compute_obb."""

    def test_compute_obb_box(self) -> None:
        box = trimesh.creation.box(extents=[2.0, 4.0, 6.0])
        obb = compute_obb(box)

        assert obb.corners.shape == (8, 3)
        assert obb.extents.shape == (3,)
        assert obb.to_origin is not None and obb.to_origin.shape == (4, 4)
        # Sorted extents should match box dimensions
        assert np.allclose(sorted(obb.extents), [2.0, 4.0, 6.0], atol=1e-4)


class TestPrepareMeshBuffers:
    """Test suite for prepare_mesh_buffers."""

    def test_prepare_untextured_mesh(self, sample_box_mesh: trimesh.Trimesh) -> None:
        buf = prepare_mesh_buffers(sample_box_mesh)
        assert isinstance(buf, MeshBuffers)
        assert buf.vertices.shape == (len(sample_box_mesh.vertices), 3)
        assert buf.faces.shape == (len(sample_box_mesh.faces), 3)
        assert buf.normals.shape == (len(sample_box_mesh.vertex_normals), 3)
        assert buf.vertex_colors is not None
        assert buf.vertex_colors.shape == (len(sample_box_mesh.vertices), 3)
        assert buf.texture is None

    def test_prepare_textured_mesh(self) -> None:
        # Create a textured mesh with visual material and UVs
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        uv = np.random.rand(len(mesh.vertices), 2).astype(np.float32)
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=uv,
            image=img,
        )

        buf = prepare_mesh_buffers(mesh, max_tex_size=16, flip_uv=True)
        assert buf.texture is not None
        assert buf.texture.shape[1] <= 16
        assert buf.texture.shape[2] <= 16
        assert buf.uvs is not None
        assert buf.uvs.shape == (len(mesh.vertices), 2)
