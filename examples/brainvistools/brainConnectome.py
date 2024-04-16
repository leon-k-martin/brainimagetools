#!/usr/bin/env python
# coding: utf-8
"""
A Connectome inside the brain
=============================

Let's plot a connectivity matrix with actual ROI coordinates.
"""

from brainvistools.networks import connectome
import numpy as np
import matplotlib.pyplot as plt

matrix = np.random.rand(379, 379)
plt.imshow(matrix)

connectome.plot_brain_connectome(matrix)
