#!/usr/bin/env python
# coding: utf-8
"""
Custom colormaps
================

Some utilty functions to create own colormaps.
"""

from brainvistools.colormap import cmap_from_list, get_continuous_cmap, double_cmap

"""
Discrete colormaps
------------------
"""
color_list = [(0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1)]
cmap = cmap_from_list(color_list, name="my_cmap")
cmap

"""
Continous colormaps
-------------------
"""
hex_list = ["#0000FF", "#00FF00", "#FF0000"]
cmap = get_continuous_cmap(hex_list)
cmap

"""
Mirrored colormaps
------------------
"""
cmap = double_cmap(cmap)
cmap
