
import numpy as np
import nibabel as nib
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid
from brainspace.mesh import mesh_creation
from brainspace import plotting
from brainspace.mesh import mesh_io


# Generate a level set about zero of two identical ellipsoids in 3D
img = nib.load('/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface.nii.gz')
print(img)

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(img.get_fdata(), level=0)
mesh_poly = mesh_creation.build_polydata(verts, faces)
mesh_io.write_surface(mesh_poly, '/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface.surf.gii', otype='gii')
surface = mesh_io.read_surface('/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface.surf.gii')
print(surface)
# print(verts.shape)
# plt.imshow(verts[:,:,250])
# plt.show()

plotting.surface_plotting.plot_surf({1:mesh_poly}, layout=(1,1))
