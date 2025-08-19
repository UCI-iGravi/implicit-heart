# Standard libraries for parsing.
import os
import glob

# Libraries for data and neural network models
import numpy as np
#import nibabel as nib

# Libraries for visualization
import pyvista as pv

#curr_data = 'data/mmwhs/MRI/mr_train/mr_train_1020_label.nii.gz'
#nib_data = nib.load(curr_data)
#nib_spacing = nib_data.header.get_zooms()
#print('Spacing:', nib_spacing)



# (10.0, 1.5625, 1.5625) - MICCAI 2017
# (1.05, 0.9722222, 0.9722222) - MMWHS 1020
npy_list = ['1.0', '3.0', '4.0', '5.0', 
            '6.0', '7.0', '8.0', '9.0',
            '11.0', '12.0', '13.0', '14.0',
            '15.0', '16.0']
#npy_list = ['205']
spacing = (1.05, 0.9722222, 0.9722222)
folder_path = 'output/cvat/'

# Setup plot
pl = pv.Plotter()

# Create all_vol if needed
create_all_vol = False
if create_all_vol:
    for i in range(len(npy_list)):
        curr_label = str(npy_list[i])
        curr_filepath = folder_path + '/' + curr_label + '.npy'
        curr_volume = np.load(curr_filepath)
        
        if i == 0:
            all_vol = np.zeros(curr_volume.shape)

        # Manually fix label 205 on MR 1020
        #if curr_label == '205':
        #    curr_volume[61] = np.zeros((curr_volume.shape[1], curr_volume.shape[2]))
            
        #curr_volume = curr_volume[:, :144, :]

        # Setup volume actor
        curr_volume = curr_volume.astype(np.uint8)
        curr_volume[curr_volume != 0] = int(float(curr_label)) # 170 default
        
        # Get indices where original array is zero
        target_indices = np.where(curr_volume != 0)
        # Replace zero values with new array values
        all_vol[target_indices] = curr_volume[target_indices]
        #all_vol += curr_volume    
    np.save(folder_path + 'all_vol2.npy', all_vol)
else:
    all_vol = np.load(folder_path + 'all_vol2.npy')
all_vol = pv.wrap(all_vol)
all_vol.spacing = spacing
vol_actor = pl.add_volume(all_vol, cmap="glasbey", opacity="foreground", shade=True)
#vol_actor = pl.add_mesh(all_vol.extract_geometry(), cmap="glasbey", opacity='sigmoid_6')

#pl.add_volume_clip_plane(vol_actor, normal='z')
#pl.add_mesh_slice(all_vol, normal='z', cmap='glasbey')

pl.add_text(folder_path, position='upper_edge', color='white', shadow=True, font_size=18)
pl.remove_scalar_bar()
pl.set_background('black')
pl.camera.zoom(2.5)
pl.camera_position = 'xz'
pl.show()