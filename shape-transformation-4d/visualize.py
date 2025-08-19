# Standard libraries for parsing.
import argparse
import os
import glob

# Libraries for data and neural network models
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Libraries for visualization
import vtk
import pyvista as pv
from pyvista import examples


# My libraries
from utils.data_loading import load_image

def get_args() -> argparse.ArgumentParser:
	"""
	Retrieve arguments from running python visualize.py
	:return: argument object with argument values
	"""
	parser = argparse.ArgumentParser(description='Creates a visualization of segmentation masks inside a folder path')
	parser.add_argument('input')
	return parser.parse_args()


def load_directory(folder: str) -> [np.array]:
	"""
	Given a folder directory containing masks, load all of the masks in order and 
	create a volume from them.
	:param folder: the folder path to load from
	:return: returns a list of ordered volumes of the segmentation masks
	"""
	# Check if there are any files mistakenly left in the base directory.
	directory_contents = os.listdir(folder)
	for c in directory_contents:
		if os.path.isfile(folder + "/" + c):
			raise Exception("Error! Do not have any masks in the base directory. \
							 Move them into individual folders for each frame! \
							 If you are working with a static set of masks, move them into one folder!")
	 
	# Sort the folders in alphabetical order for each frame
	folders = sorted([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])
	volume_list = []
	
	# Load up each segmentation volume and save it to a list of volumes. Each index in volume_list is a frame.
	for f in folders:
		curr_path = folder + "/" + f + "/*"
		files = sorted(glob.glob(curr_path))
		segmentations = np.array([np.asarray(load_image(img)) for img in files])
		volume_list.append(segmentations)
	return volume_list


def main(args: argparse.ArgumentParser, volumes: [np.array]):
	"""
	Main function where most of the processing and visualization code lies.
	:param args: arguments from running python visualize.py
	:param volumes: a set of 3D volumes of segmentation masks that we want to build a 3D visualization of
	:return:
	"""
	print(volumes[0].shape)
	print(volumes[1].shape)
	print()
	# https://docs.pyvista.org/examples/02-plot/gif.html
	# https://docs.pyvista.org/examples/02-plot/movie.html
	
	pl = pv.Plotter()
	#v1 = np.where(volumes[0] == 1, 255, volumes[0])
	#v2 = np.where(volumes[1] == 1, 127, volumes[1])
	v0 = volumes[0]
	v1 = volumes[1]
	pl.add_volume(v0, cmap="viridis", opacity="sigmoid_6", shade=True)
	#pl.add_volume(v2, cmap="viridis", opacity="sigmoid_6", shade=True)
	pl.camera.zoom(2.5)
	pl.show()
	return


def animate(args: argparse.ArgumentParser, volumes: [np.array], out_path: str):
	"""
	Generate a .gif animation of each frame in the list of volumes.
	:param args: arguments from running python visualize.py
	:param volumes: a set of 3D volumes of segmentation masks that we want to build a 3D visualization of
	:param out_path: the filepath to output the .gif file
	"""
	plotter = pv.Plotter(notebook=False, off_screen=True)
	plotter.open_gif(out_path)
	
	has_zoom = False
	
	# Go through each frame, write it, and clear the plot.
	for v in volumes:
		#v1 = np.where(v == 1, 255, v)
		plotter.add_volume(v, cmap="viridis", opacity="sigmoid_6", shade=False)
		if not has_zoom:
			plotter.camera.zoom(2.5)
			has_zoom = True
		plotter.write_frame()
		plotter.clear()
	plotter.close()
	return


# Run this if python visualize.py is executed.
# python visualize.py run/viz/output/patient001_4d_LV/
if __name__ == '__main__':
	args = get_args()
	volumes = load_directory(args.input)
	animate(args, volumes, "out.gif")
	main(args, volumes)
