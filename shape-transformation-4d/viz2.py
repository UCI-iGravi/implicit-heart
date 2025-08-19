# Standard libraries for parsing.
import os
import glob

# Libraries for data and neural network models
import numpy as np

# Libraries for visualization
import pyvista as pv
#pv.set_jupyter_backend('ipygany')

# My libraries
v = np.load("data/npy/mmwhs.npy")
# https://docs.pyvista.org/examples/02-plot/gif.html
# https://docs.pyvista.org/examples/02-plot/movie.html

pl = pv.Plotter()
v = np.where(v == 1, 255, v)
#v2 = np.where(volumes[1] == 1, 127, volumes[1])
#v0 = volumes[0]
#v1 = volumes[1]
pl.add_volume(v, cmap="viridis", opacity="sigmoid_6", shade=True)
#pl.add_volume(v2, cmap="viridis", opacity="sigmoid_6", shade=True)
pl.camera.zoom(2.5)
pl.show()