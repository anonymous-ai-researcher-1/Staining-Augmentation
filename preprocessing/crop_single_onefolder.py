import openslide as slide
from PIL import Image
import numpy as np
from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
from os import listdir, mkdir, path, makedirs
from os.path import join 
import os
import time, sys, warnings, glob
import threading
from tqdm import tqdm
import argparse
import scipy
warnings.simplefilter('ignore')
import cv2
import imageio

Image.MAX_IMAGE_PIXELS = None

def thres_saturation(img, t=15):
	# typical t = 15
	img = rgb2hsv(img)
	h, w, c = img.shape
	sat_img = img[:, :, 1]
	sat_img = img_as_ubyte(sat_img)
	ave_sat = np.sum(sat_img) / (h * w)
	return ave_sat >= t


def crop_slide(img, img_name, img_mask, save_slide_path, save_slide_mask_path,position=(0, 0), step=(0, 0), patch_size=512): # position given as (x, y) 
	_img = img.read_region((position[0] * 4, position[1] * 4), 1, (patch_size, patch_size))
	_img = np.array(_img)[..., :3]
	if thres_saturation(_img, 15):
		patch_name = "{}_{}".format(step[0], step[1])
		imageio.imwrite(join(save_slide_path, img_name+"_"+patch_name + ".jpg"),_img)
		# io.imsave(join(save_slide_path, img_name+"_"+patch_name + ".jpg"), img_as_ubyte(img))
		img_mask_res = img_mask.crop((position[0] * 4, position[1] * 4, position[0] * 4 + patch_size, position[1] * 4 + patch_size)).convert('RGB')
		# img_mask_res = img_mask[position[0] * 4 : position[0] * 4 + patch_size, position[1] * 4 : position[1] * 4 + patch_size]
		img_mask_res = np.array(img_mask_res)[..., :3]
		# io.imsave(join(save_slide_mask_path, img_name+patch_name + ".jpg"), img_as_ubyte(img_mask * 255))       #只有边框就不成255
		# io.imsave(join(save_slide_mask_path, img_name+"_"+patch_name + ".jpg"), img_as_ubyte(img_mask))       #只有边框就不成255
		# io.imsave(join(save_slide_mask_path, img_name+"_"+patch_name + ".jpg"), img_mask)       #只有边框就不成255
		imageio.imwrite(join(save_slide_mask_path, img_name+"_"+patch_name + ".jpg"), img_mask_res)       #只有边框就不成255
		# io.imsave(join(save_slide_mask_path, img_name+"_mask_"+patch_name + ".jpg"), img_as_ubyte(img_mask * 255))



def slide_to_patch(out_base, img_slides, step):
	makedirs(out_base, exist_ok=True)
	patch_size = 512
	step_size = step * 1
	for s in range(len(img_slides)):
		img_slide = img_slides[s]
		img_name = img_slide.split(path.sep)[-1].split('.')[0]
		bag_path = join(out_base, 'imgs')
		# bag_path = join(out_base, img_name)
		bag_mask_path = join(out_base, 'masks')
		# bag_mask_path = join(out_base, img_name+'_mask')
		makedirs(bag_path, exist_ok=True)
		makedirs(bag_mask_path, exist_ok=True)
		img = slide.OpenSlide(img_slide)
		# img_mask=slide.OpenSlide(img_slide[:-4]+'_mask.tif')
		img_mask = Image.open(img_slide[:-4]+'.tif')
		# _img_mask = cv2.imread(img_slide[:-4]+'.tif')
		dimension = img.level_dimensions[0] # given as width, height
		# _img_mask = io.imread(img_slide[:-4]+'.tif') *255
		# img_mask = (transform.resize(_img_mask, dimension, order=0)).astype(np.uint8)
		# img_mask[img_mask > 0] = 255
		# if folder=='test':
		# thumbnail = np.array(img.get_thumbnail((int(dimension[0])/7, int(dimension[1])/7)))[..., :3]
		# smallimg = scipy.ndimage.zoom(img_mask, [0.1, 0.1], order = 0)
		thumbnail = np.array(img.get_thumbnail((int(dimension[0])/10, int(dimension[1])/10)))[..., :3]
		smallimg = scipy.ndimage.zoom(img_mask, [0.1, 0.1], order = 0)
		io.imsave(join(out_base, 'thumbnails', img_name + ".png"), img_as_ubyte(thumbnail))
		io.imsave(join(out_base, 'thumbnails', img_name + "_mask.png"), img_as_ubyte(smallimg))      #只有边框就不成255
		# # else:
		# 	# thumbnail = np.array(img.get_thumbnail((int(dimension[0])/28, int(dimension[1])/28)))[..., :3]
		# io.imsave(join(out_base, 'thumbnails', img_name + ".png"), img_as_ubyte(thumbnail))
		# # io.imsave(join(out_base, 'thumbnails', img_name + "_mask.png"), img_as_ubyte(smallimg * 255))      #只有边框就不成255
		# io.imsave(join(out_base, 'thumbnails', img_name + "_mask.png"), img_as_ubyte(smallimg))      #只有边框就不成255
		step_y_max = int(np.floor(dimension[1]/step_size)) # rows
		step_x_max = int(np.floor(dimension[0]/step_size)) # columns
		for j in range(step_y_max): # rows
			for i in range(step_x_max): # columns
				crop_slide(img, img_name, img_mask, bag_path, bag_mask_path, (i*step_size, j*step_size), step=(j, i), patch_size=patch_size)
				# crop_slide_mask(img_mask, bag_mask_path, (i*step_size, j*step_size), step=(j, i), patch_size=patch_size)
			sys.stdout.write('\r Cropped: {}/{} -- {}/{}'.format(s+1, len(img_slides), j+1, step_y_max))




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate patches from testing slides')
	parser.add_argument('--dataset', type=str, default='tcga', help='Dataset name [tcga]')
	args = parser.parse_args()
	path_base = ('/home/sdb/huangruiwei/data/PDL1_2')
	out_base = ('/home/sdb/huangruiwei/data/PDL1_patch3')
	# folder = '/home/sda/huangruiwei/data/slides666'
	makedirs('/home/sdb/huangruiwei/data/PDL1_patch3/thumbnails', exist_ok=True)
	all_slides = glob.glob(join(path_base, '*.svs')) 
	#新增  是否存在mask检测
	for _slide in all_slides:
		if(os.path.exists(join(path_base,_slide[:-3]+'tif'))):
			# print(_slide[:-3])
			pass
		else:
			all_slides.remove(_slide)
	# all_slides = glob.glob(join(path_base, '*.svs')) + glob.glob(join(path_base, '*.tif'))
	parser.add_argument('--overlap', type=int, default=0)
	parser.add_argument('--patch_size', type=int, default=512)
	args = parser.parse_args()
	print('Cropping patches, please be patient')
	step = args.patch_size - args.overlap
	slide_to_patch(out_base, all_slides, step)
	print('\ndone!')



