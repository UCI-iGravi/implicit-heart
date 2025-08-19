import pyvista as pv
import numpy as np

i0 = np.load('data/interp/M0.npy')
i1 = np.load('data/interp/M15.npy')
i05 = np.load('data/interp/interpolated3dlv2.npy')


curr_volume = i1
out_path = 'interp1.gif'
title = '1'


curr_volume = curr_volume.astype(np.uint8)
curr_volume[curr_volume != 0] = 170
pl = pv.Plotter(notebook=False, off_screen=True)
pl.open_gif(out_path)

vol = pv.wrap(curr_volume)
vol.spacing = (10.0, 1.5625, 1.5625)
pl.add_volume(vol, cmap="Reds", opacity="sigmoid_6", shade=True)
pl.remove_scalar_bar()

pl.add_text(title, position='upper_edge', color='black', shadow=True, font_size=18)


pl.camera.zoom(2.5)
pl.write_frame()
pl.clear()
pl.close()