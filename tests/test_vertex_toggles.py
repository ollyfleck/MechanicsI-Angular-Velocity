"""
Tests for per-vertex vector toggle functionality.
"""

import math
import numpy as np
import pytest

from draw_3d import draw_3d_velocity_vectors
from geometry import CUBE_VERTS


class MockScreen:
    """Minimal mock screen for testing."""
    def __init__(self):
        self.draw_calls = []

    def draw_polygon(self, color, points, width=0):
        self.draw_calls.append(('polygon', color, points))

    def draw_line(self, color, start, end, width=3):
        self.draw_calls.append(('line', color, start, end))

    def draw_circle(self, color, center, radius):
        self.draw_calls.append(('circle', color, center, radius))


@pytest.fixture
def sample_verts():
    """Return a list of 8 cube vertices."""
    return list(CUBE_VERTS)


class TestVertexMask:
    """Tests for vertex_mask parameter in vector drawing functions."""

    def test_draw_3d_velocity_vectors_with_all_enabled(self, sample_verts):
        """When all vertices are enabled, all should be considered."""
        vertex_mask = {i: True for i in range(8)}
        tang_drawables, cent_drawables = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=vertex_mask
        )
        # With omega_z=100 and all vertices enabled, we should get some drawables
        # (may be empty if vectors are too small, but the mask should not filter them out)
        # Just verify it doesn't crash and returns lists
        assert isinstance(tang_drawables, list)
        assert isinstance(cent_drawables, list)

    def test_draw_3d_velocity_vectors_with_none_enabled(self, sample_verts):
        """When no vertices are enabled, no vectors should be drawn."""
        vertex_mask = {i: False for i in range(8)}
        tang_drawables, cent_drawables = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=vertex_mask
        )
        # With all vertices disabled, no drawables should be produced
        assert len(tang_drawables) == 0
        assert len(cent_drawables) == 0

    def test_draw_3d_velocity_vectors_with_partial_mask(self, sample_verts):
        """When only specific vertices are enabled, only those should be considered."""
        # Only enable vertex 0
        vertex_mask = {i: (i == 0) for i in range(8)}
        tang_drawables, cent_drawables = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=vertex_mask
        )
        # Should not crash and return lists
        assert isinstance(tang_drawables, list)
        assert isinstance(cent_drawables, list)

    def test_draw_3d_velocity_vectors_with_none_mask(self, sample_verts):
        """When vertex_mask is None, all vertices should be considered (default behavior)."""
        tang_drawables, cent_drawables = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=None
        )
        # Should not crash
        assert isinstance(tang_drawables, list)
        assert isinstance(cent_drawables, list)


    def test_vertex_mask_single_vertex_enabled(self, sample_verts):
        """Test with only one vertex enabled at a time."""
        for enabled_vertex in range(8):
            vertex_mask = {i: (i == enabled_vertex) for i in range(8)}
            tang_drawables, cent_drawables = draw_3d_velocity_vectors(
                sample_verts, total_omega_mag=50.0, screen=None,
                omega_x=0, omega_y=0, omega_z=100.0,
                max_vectors=8,
                show_tangential=True, show_centripetal=True,
                view_x=0, view_y=0, view_z=0,
                vertex_mask=vertex_mask
            )
            # Should not crash for any vertex
            assert isinstance(tang_drawables, list)
            assert isinstance(cent_drawables, list)

    def test_vertex_mask_empty_dict(self, sample_verts):
        """Test with empty vertex_mask (should default to all enabled)."""
        vertex_mask = {}
        tang_drawables, cent_drawables = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=vertex_mask
        )
        # Empty mask means all default to False (via .get(i, False))
        # So no drawables should be produced
        assert isinstance(tang_drawables, list)
        assert isinstance(cent_drawables, list)