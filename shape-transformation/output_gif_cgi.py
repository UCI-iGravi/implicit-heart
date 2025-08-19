import pyvista as pv
import numpy as np
import edt

pv.global_theme.anti_aliasing = 'msaa'
pv.global_theme.multi_samples = 16

# Settings and paths
i0 = np.load('output/anim/M0.npy')
i1 = np.load('output/anim/M11.npy')
iinterp = np.load('output/anim/gravi-M0-M11-interp.npy')
out_path = 'output/anim/interp_LV-'

# Setup and iterate through volumes to make gif
vol_list = [i0, iinterp, i1]
title_list = ['0', '0.5', '1']
for i in range(len(vol_list)):
    title = title_list[i]
    curr_volume = vol_list[i].copy()
    curr_volume = curr_volume.astype(np.uint8)
    curr_volume[curr_volume != 0] = 170
    
    
        
    pl = pv.Plotter(notebook=False, off_screen=True)
    pl.open_gif(out_path + str(i) + '.gif')
    
    """
    if i == 0:
        prev_volume = vol_list[0].copy()
        prev_volume = prev_volume.astype(np.uint8)
        prev_volume[prev_volume != 0] = 170
        
        prev_volume = edt.edt(prev_volume).astype(float)
        #prev_volume[prev_volume != 0] *= 100.0
        prev_volume[prev_volume != 0] = 1
        
        
        prev_vol = pv.wrap(prev_volume)
        prev_vol.spacing = (10.0, 1.5625, 1.5625)
        #pl.add_mesh(prev_vol.extract_all_edges(), color='k')
        pl.add_volume(prev_vol, cmap='Greens', opacity="foreground", shade=False)
    """
    
    if i != 0:
        prev_volume = vol_list[i-1].copy()
        prev_volume = prev_volume.astype(np.uint8)
        prev_volume[prev_volume != 0] = 170
        
        prev_volume = edt.edt(prev_volume).astype(float)
        #prev_volume[prev_volume != 0] *= 100.0
        prev_volume[prev_volume != 0] = 1
        
        
        prev_vol = pv.wrap(prev_volume)
        prev_vol.spacing = (10.0, 1.5625, 1.5625)
        #pl.add_mesh(prev_vol.extract_all_edges(), color='k')
        pl.add_volume(prev_vol, cmap='Greens', opacity="foreground", shade=False)
        #pl.add_volume(prev_vol, cmap='viridis', opacity="linear", shade=False)
        
    curr_volume = edt.edt(curr_volume).astype(float)
    #for i in range(0, curr_volume.shape[0]):
    #    for j in range(0, curr_volume.shape[1]):
    #        for k in range(0, curr_volume.shape[2]):
    #            if curr_volume[i][j][k] != 0:
    #                curr_volume[i][j][k] = curr_volume[i][j][k] * (1/curr_volume[i][j][k])
    
    curr_volume[curr_volume != 0] *= 100.0
    #curr_volume[curr_volume <= 0] = -100
    #print(np.max(curr_volume))
    #print(np.min(curr_volume))
    #print(np.mean(curr_volume))
    
    
    vol = pv.wrap(curr_volume)
    vol.spacing = (10.0, 1.5625, 1.5625)
    
    
    # Main LV
    pl.add_volume(vol, cmap="hot", opacity=[0, 0.8, 0.5, 0.6, 0.1], shade=False)  # sigmoid_6, Reds
    
    # Overlay
    # Try 0 0.2 0.6 0.2 0.1
    pl.add_volume(vol, cmap="hot", opacity=[0, 0.2, 0.1, 0.2, 0.1], shade=False)  # sigmoid_6, Reds
    
    #pl.add_volume(vol, cmap="Reds", opacity=[0, 0, 0.2, 0.3, 0], shade=False)  # sigmoid_6, Reds
    pl.remove_scalar_bar()

    pl.add_text(title, position='upper_edge', color='white', 
                shadow=True, font_size=18)

    #pl.set_position([45.0, 167.96875, 1220.92255])
    pl.camera.zoom(4) # 2.5
    pl.set_background('black')
    pl.write_frame()
    pl.clear()
    
pl.close()