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
        """When all vertices are enabled, vectors should be produced for vertices with r >= 0.5."""
        vertex_mask = {i: True for i in range(8)}
        tang_drawables, cent_drawables = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=vertex_mask
        )
        assert isinstance(tang_drawables, list)
        assert isinstance(cent_drawables, list)
        # With omega_z=100 and vertices at distance > 0.5, we should get drawables
        total_drawables = len(tang_drawables) + len(cent_drawables)
        assert total_drawables > 0, "Should produce drawables when vertices are enabled"

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
        assert len(tang_drawables) == 0, "Should produce no tangential drawables when all disabled"
        assert len(cent_drawables) == 0, "Should produce no centripetal drawables when all disabled"

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
        # With only 1 vertex enabled, should produce fewer drawables than all enabled
        assert isinstance(tang_drawables, list)
        assert isinstance(cent_drawables, list)
        # Compare with all-enabled mask to verify mask filtering works
        tang_all, cent_all = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask={i: True for i in range(8)}
        )
        total_partial = len(tang_drawables) + len(cent_drawables)
        total_all = len(tang_all) + len(cent_all)
        assert total_partial < total_all, \
            f"Partial mask ({total_partial}) should produce fewer drawables than all-enabled ({total_all})"

    def test_draw_3d_velocity_vectors_with_none_mask(self, sample_verts):
        """When vertex_mask is None, all vertices should be considered (default behavior)."""
        tang_drawables_none, cent_drawables_none = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=None
        )
        # Compare with all-enabled mask
        tang_drawables_all, cent_drawables_all = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask={i: True for i in range(8)}
        )
        # None mask should produce same number of drawables as all-enabled
        assert len(tang_drawables_none) == len(tang_drawables_all), \
            f"None mask should match all-enabled: tangential {len(tang_drawables_none)} vs {len(tang_drawables_all)}"
        assert len(cent_drawables_none) == len(cent_drawables_all), \
            f"None mask should match all-enabled: centripetal {len(cent_drawables_none)} vs {len(cent_drawables_all)}"

    def test_vertex_mask_single_vertex_enabled(self, sample_verts):
        """Test with only one vertex enabled at a time."""
        # First get the all-enabled baseline
        tang_all, cent_all = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask={i: True for i in range(8)}
        )
        total_all = len(tang_all) + len(cent_all)
        
        # Now test each single-vertex mask
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
            total_single = len(tang_drawables) + len(cent_drawables)
            # Each single vertex should produce fewer drawables than all enabled
            assert total_single < total_all, \
                f"Vertex {enabled_vertex}: single mask ({total_single}) should produce fewer drawables than all-enabled ({total_all})"

    def test_vertex_mask_empty_dict_behavior(self, sample_verts):
        """Test with empty vertex_mask - should result in no drawables (default False)."""
        vertex_mask = {}
        tang_drawables, cent_drawables = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=vertex_mask
        )
        # Empty dict means all default to False (via .get(i, False))
        # So no drawables should be produced
        assert len(tang_drawables) == 0, \
            f"Empty mask should produce no drawables, got {len(tang_drawables)} tangential"
        assert len(cent_drawables) == 0, \
            f"Empty mask should produce no drawables, got {len(cent_drawables)} centripetal"

    def test_vertex_mask_all_vs_none_produces_different_results(self, sample_verts):
        """Test that all-enabled produces more drawables than none-enabled."""
        vertex_mask_all = {i: True for i in range(8)}
        tang_all, cent_all = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=vertex_mask_all
        )

        vertex_mask_none = {i: False for i in range(8)}
        tang_none, cent_none = draw_3d_velocity_vectors(
            sample_verts, total_omega_mag=50.0, screen=None,
            omega_x=0, omega_y=0, omega_z=100.0,
            max_vectors=8,
            show_tangential=True, show_centripetal=True,
            view_x=0, view_y=0, view_z=0,
            vertex_mask=vertex_mask_none
        )

        total_all = len(tang_all) + len(cent_all)
        total_none = len(tang_none) + len(cent_none)

        assert total_all > total_none, \
            f"All-enabled ({total_all}) should produce more drawables than none-enabled ({total_none})"