"""
Configuration module - loads and provides access to all configuration values.
"""

import os
import yaml

# Load configuration from YAML file relative to this module's location
config_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(config_dir, 'config.yaml')
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

# ==================== CONSTANTS ====================

SCREEN_W = CONFIG['screen']['width']
SCREEN_H = CONFIG['screen']['height']
CUBE_SIZE = CONFIG['cube_size']
FRAME_RATE = CONFIG['physics'].get('frame_rate', 60)

# ==================== COLORS ====================

COLORS = {
    'bg': tuple(CONFIG['colors']['bg']),
    'cube_edges': tuple(CONFIG['colors']['cube_edges']),
    'omega': tuple(CONFIG['colors']['omega']),
    'velocity': tuple(CONFIG['colors']['velocity']),
    'axis': tuple(CONFIG['colors']['axis']),
    'text_main': tuple(CONFIG['colors']['text_main']),
    'text_sub': tuple(CONFIG['colors']['text_sub']),
}

# Handle highlight color fallback if not in config
if 'highlight' in CONFIG['colors']:
    COLORS['highlight'] = tuple(CONFIG['colors']['highlight'])
else:
    COLORS['highlight'] = (255, 220, 180)

# ==================== VECTOR CONFIGS ====================


def get_vector_config():
    """Get vector visualization configuration, falling back to legacy keys."""
    vec_vis = CONFIG.get('vector_visualization', {})

    # Omega vector config
    omega_cfg = vec_vis.get('omega', {})
    if not omega_cfg:
        omega_cfg = {
            'length_at_max': 60,
            'min_length': 30,
            'enabled': True
        }

    # Tangential velocity config
    tang_cfg = vec_vis.get('tangential_velocity', {})
    if not tang_cfg:
        tang_cfg = {
            'scale_factor': CONFIG.get('vector_scales', {}).get('tangential', 0.08),
            'max_length': CONFIG.get('vector_scales', {}).get('tangential_max', 8),
            'min_length': 0,
            'enabled': True
        }

    # Centripetal acceleration config
    cent_cfg = vec_vis.get('centripetal_acceleration', {})
    if not cent_cfg:
        cent_cfg = {
            'length_at_max': CONFIG.get('vector_scales', {}).get('normal_vertex', 6),
            'min_length': 0,
            'enabled': True
        }

    # Face normals config
    face_cfg = vec_vis.get('face_normals', {})
    if not face_cfg:
        face_cfg = {
            'fixed_length': CONFIG.get('vector_scales', {}).get('normal_face', 25),
            'min_length': CONFIG.get('vector_scales', {}).get('normal_face', 25),
            'enabled': True
        }

    return omega_cfg, tang_cfg, cent_cfg, face_cfg


VECTOR_CONFIGS = get_vector_config()