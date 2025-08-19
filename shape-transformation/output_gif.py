import pyvista as pv
import numpy as np
import glob

# Settings and paths
anim_folder = "output/anim/full_4d_interp/"
anim_files = sorted(glob.glob(anim_folder + '/*.npy'))
out_path = 'output/anim/full_4d_interp/gif/'

pl = pv.Plotter(lighting='three lights')
pl.set_background('black')

#light = pv.Light()
#light.set_direction_angle(30, 20)
#pl.add_light(light)

VIEW = 0
if VIEW:
    pl.open_gif(out_path + '/4dlv_side.gif')
else:
    pl.open_gif(out_path + '/4dlv.gif')
    
curr_text = 0
for i in range(len(anim_files)):
    curr_volume = np.load(anim_files[i])
    curr_volume = curr_volume.astype(np.uint8)
    curr_volume[curr_volume != 0] = 255
        
    vol = pv.wrap(curr_volume)
    vol.spacing = (10.0, 1.5625, 1.5625)
    lv_actor1 = pl.add_volume(vol, cmap="Reds", opacity="foreground", shade=True)
    #lv_actor2 = pl.add_mesh(vol, cmap="Reds", opacity='foreground')
    pl.remove_scalar_bar()

    text = pl.add_text(str(curr_text), position='upper_edge', color='white', 
                shadow=True, font_size=18)

    if VIEW:
        pl.set_position([45.0, 167.96875, 1220.92255])
    pl.write_frame()
    #pl.clear()
    pl.remove_actor(lv_actor1)
    #pl.remove_actor(lv_actor2)
    pl.remove_actor(text)
    
    curr_text += 0.5
    
pl.close()