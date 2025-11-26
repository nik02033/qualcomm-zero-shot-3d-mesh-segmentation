#!/usr/bin/env python3
"""
sumparts_config.py
Utility file to parse and store SUM Parts labels
"""

# SUM-Parts canonical class definitions
SUMPARTS_CLASSES = {
    # Face labels (0-12)
    0: "unclassified",
    1: "terrain",
    2: "high_vegetation",
    3: "facade_surface",
    4: "water",
    5: "car",
    6: "boat",
    7: "roof_surface",
    8: "chimney",
    9: "dormer",
    10: "balcony",
    11: "roof_installation",
    12: "wall",
    
    # Texture labels (13-20)
    13: "window",
    14: "door",
    15: "low_vegetation",
    16: "impervious_surface",
    17: "road",
    18: "road_marking",
    19: "cycle_lane",
    20: "sidewalk"
}

# Inverted mapping: class_name -> class_id
SUMPARTS_NAME_TO_ID = {name.lower(): cid for cid, name in SUMPARTS_CLASSES.items()}

# Number of face label classes (0-12)
NUM_FACE_CLASSES = 13

# Number of texture label classes (13-20)
NUM_TEXTURE_CLASSES = 8

# Total classes
NUM_TOTAL_CLASSES = 21

# Classes to ignore in evaluation (typically unclassified)
DEFAULT_IGNORE_CLASS = 0

# Color palette for visualization (RGB, 0-1 range)
CLASS_COLORS = {
    0: (0.60, 0.60, 0.60),  # unclassified - gray
    1: (0.55, 0.40, 0.25),  # terrain - brown
    2: (0.10, 0.60, 0.10),  # high_vegetation - dark green
    3: (0.90, 0.80, 0.70),  # facade_surface - beige
    4: (0.10, 0.40, 0.90),  # water - blue
    5: (0.90, 0.10, 0.10),  # car - red
    6: (0.10, 0.10, 0.90),  # boat - dark blue
    7: (0.80, 0.40, 0.20),  # roof_surface - terracotta
    8: (0.50, 0.10, 0.10),  # chimney - dark red
    9: (0.70, 0.50, 0.30),  # dormer - brown
    10: (0.60, 0.60, 0.80),  # balcony - light purple
    11: (0.40, 0.20, 0.10),  # roof_installation - dark brown
    12: (0.85, 0.85, 0.85),  # wall - light gray
    13: (0.39, 0.39, 1.00),  # window - blue
    14: (0.59, 0.12, 0.24),  # door - burgundy
    15: (0.78, 1.00, 0.00),  # low_vegetation - lime
    16: (0.39, 0.59, 0.59),  # impervious_surface - teal
    17: (0.78, 0.78, 0.78),  # road - light gray
    18: (0.59, 0.39, 0.59),  # road_marking - purple
    19: (1.00, 0.33, 0.50),  # cycle_lane - pink
    20: (1.00, 1.00, 0.67),  # sidewalk - pale yellow
}


def get_class_name(class_id: int) -> str:
    """Get canonical class name from ID."""
    return SUMPARTS_CLASSES.get(class_id, f"class_{class_id}")


def get_class_id(class_name: str) -> int:
    """Get class ID from name (case-insensitive)."""
    return SUMPARTS_NAME_TO_ID.get(class_name.lower(), -1)


def load_label_mapping(json_path: str) -> dict:
    """
    Load label mapping from JSON file.
    
    Args:
        json_path: Path to label_mapping.json
    
    Returns:
        Dictionary mapping RAM++ labels to SUM-Parts class IDs
    """
    import json
    from pathlib import Path
    
    if not Path(json_path).exists():
        print(f"[WARN] Label mapping not found: {json_path}")
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mappings = data.get("mappings", {})
    
    # Convert to class IDs
    id_mappings = {}
    for ram_label, sumparts_name in mappings.items():
        sumparts_id = get_class_id(sumparts_name)
        if sumparts_id >= 0:
            id_mappings[ram_label.lower()] = sumparts_id
        else:
            print(f"[WARN] Unknown SUM-Parts class: '{sumparts_name}' for RAM++ label '{ram_label}'")
    
    return id_mappings


# Export key constants
__all__ = [
    'SUMPARTS_CLASSES',
    'SUMPARTS_NAME_TO_ID',
    'NUM_FACE_CLASSES',
    'NUM_TOTAL_CLASSES',
    'DEFAULT_IGNORE_CLASS',
    'CLASS_COLORS',
    'get_class_name',
    'get_class_id',
    'load_label_mapping',
]