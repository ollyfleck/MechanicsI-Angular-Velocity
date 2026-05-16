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
# Access: CONFIG['vectors'][vector_name] for vector settings
# Access: CONFIG['vectors'][vector_name]['geometry'] for geometry settings
# Access: CONFIG['vector_shading'] for global shading settings
