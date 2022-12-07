"""
Modeule  that holds satellite information, TODO: change this to a json
"""

import datetime

satellite = {
    "sentinel-2": {
        "name": "COPERNICUS/S2_SR_HARMONIZED",
        "launch": datetime.date(2017, 3, 8),
        "rgb_vis_params": {
            "bands": ["B4", "B3", "B2"],
            "min": 0,
            "max": 2000,
        },
        "false_color_vis_params": {
            "bands": ["B8", "B4", "B3"],
            "min": 120,
            "max": 2898,
        },
    },
    "landsat-8": {
        "name": "LANDSAT/LC08",
        "launch": datetime.date(2011, 2, 11),
        "rgb_vis_params": {
            "bands": ["B4", "B3", "B2"],
            "min": -152,
            "max": 1811,
        },
        "false_color_vis_params": {
            "bands": ["B5", "B4", "B3"],
            "min": 284,
            "max": 3584,
        },
    },
}
