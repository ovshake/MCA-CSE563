from PIL import Image , ImageFilter
from skimage.feature import peak_local_max, hessian_matrix, haar_like_feature, haar_like_feature_coord
from skimage.feature.blob import _prune_blobs
from skimage.filters import gaussian 
import numpy as np 
import math
import os 
from sys import exit
from skimage.transform import integral_image
from tqdm import tqdm
import json

def dump_json(d,filename):
	with open(filename, 'w') as fp:
		json.dump(d, fp)

def get_surf_features(img, sigma_array):
	img = np.array(img)
	all_blobs = []
	features = None 
	for sig in sigma_array:
		intgl_img = integral_image(img)
		h_rr, h_rc, h_cc = hessian_matrix(intgl_img, sigma=sig)
		det_h = np.multiply(h_cc, h_rr) - 0.81 * (h_rc ** 2)
		blobs = peak_local_max(det_h, threshold_rel = .1) 
		# print(blobs)
		# print(blobs)
		blobs = [[x,y,sig] for (x,y) in blobs]
		all_blobs.extend(blobs)
		# blobs = np.asarray(blobs)
	# pruned_blobs = _prune_blobs(np.asarray(blobs), overlap = 0.2)
	# print(pruned_blobs)
	# print('Pruning Complete')
	return all_blobs





if __name__ == '__main__':
	Q = 64
	# file_name = 'all_souls_000000.jpg'
	# im1 = Image.open(file_name)  
	# im1 = im1.resize((64,64)) 
	# im1 = im1.quantize(Q)
	# im1 = gaussian(np.array(im1)) 
	# intgl_img = integral_image(im1)
	# # print(intl_img)
	# h_rr, h_rc, h_cc = hessian_matrix(intgl_img, sigma=1)
	# det_H = np.multiply(h_cc, h_rr) - 0.81 * (h_rc ** 2)
	# blobs = peak_local_max(det_H) 
	# # print(blobs)
	# for x,y in blobs:
	# 	# haar1 = haar_like_feature(intgl_img, x,y,20,20,feature_type='type-2-x') 
	# 	# haar2 = haar_like_feature(intgl_img, x,y,20,20,feature_type='type-2-y')
	# 	haar = haar_like_feature(intgl_img, x,y,20,20,feature_type=['type-2-x','type-2-y'],feature_coord = haar_like_feature_coord(20,20,['type-2-x','type-2-y'])[0]) 
	# 	print(haar)
	images = 'images/'
	save_dir = 'blob_json_SURF/'
	imgs = os.listdir(images)
	total_files = len(imgs)
	sigma_array = [9, 15, 21, 27, 15, 27, 39, 51, 27, 51, 75, 99, 51, 99, 147, 195]
	blob_dict = {} 
	for i,img in tqdm(enumerate(imgs)):
		file_name = images + img
		fn_no_ext = img.split('.')[0] 
		# print('{}/{} Done\r'.format(i+1,total_files))
		im = Image.open(file_name)
		im = im.resize((64,64))
		im = im.quantize(Q) 
		pruned_blobs = get_surf_features(im,sigma_array) 
		# pruned_blobs = pruned_blobs.tolist() 
		pruned_blobs = [[int(x), int(y), int(z)] for (x,y,z) in pruned_blobs]
		blob_dict[fn_no_ext] = pruned_blobs

	dump_json(blob_dict,save_dir + 'blob_dict_SURF.json')





	


