#!/usr/bin/env python
# coding: utf-8
"""
Formatted Glass Brain
=========================

Wraps nilearn's plotting.plot_glass_brain() and formats it nicely.
"""

import matplotlib.pyplot as plt
import nibabel as nib

from brainvistools import constants, style, volume

style.init_style()

img = nib.load(constants.DATA_DIR + "/MESH-Cognition_tvbase.nii")

fig, ax = plt.subplot()
volume.formatted_glass_brain(img, title="TVBase-Map Cognition", ax=ax)
plt.show()
