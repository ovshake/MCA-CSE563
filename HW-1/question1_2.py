from PIL import Image , ImageFilter
from skimage.feature import peak_local_max 
from skimage.feature.blob import _prune_blobs
import numpy as np 
import math
import os 
from sys import exit
from tqdm import tqdm
eps = math.e 
pi = math.pi 
import json


def dump_json(d,filename):
	with open(filename, 'w') as fp:
		json.dump(d, fp)


def apply_gaussian_smoothing_at_T(im_array,t):
	for i,x in enumerate(im_array):
		for j,y in enumerate(x):
			im_array[i,j] = (( eps ** (- ((i+1)**2 + (j+1)**2) / 2*t)) / (2 * pi * t)) * y 

	return im_array 


def get_smoothed_image_at_T(im, T):
	im_mat = np.array(im) 
	for i,x in enumerate(im_mat):
		for j,y in enumerate(x):
			im_mat[i,j] = (((i+1)**2 + (j+1)**2 - 2*(T**2)) / pi * (T**4)) * (eps ** (- (  (i+1)**2 + (j+1)**2  ) / ( 2* (T**2) ) ) ) * y 

	return Image.fromarray(im_mat)


def get_blobs_at_T(im, T):
	im = get_smoothed_image_at_T(im, T)
	blobs = peak_local_max(np.array(im), threshold_abs = .1)
	blob_array = [[i,j,T] for (i,j) in blobs] 
	return np.array(blob_array) 



def get_all_blobs(im, T_list):
	all_blobs_with_T = np.asarray([[0,0,0]])
	# print(all_blobs_with_T.shape)
	for t in T_list:
		blobs_at_T = get_blobs_at_T(im, t)
		# print(blobs_at_T.shape)
		# all_blobs_with_T = np.concatenate((all_blobs_with_T,blobs_at_T), axis = 0)
		if blobs_at_T.shape[0] == 0:
			continue
		all_blobs_with_T = np.concatenate((all_blobs_with_T, blobs_at_T), axis = 0) 
	all_blobs_with_T = all_blobs_with_T[1:, :]

	return all_blobs_with_T


def remove_overlapping_blobs(all_blobs_with_T, overlap = 0.5):
	pruned_blobs = _prune_blobs(all_blobs_with_T, overlap = overlap)
	return pruned_blobs



if __name__ == '__main__':
	Q = 64
	blob_dict = {}
	# file_name = 'all_souls_000000.jpg'
	# im1 = Image.open(file_name)  
	# im1 = im1.resize((64,64)) 
	# im1 = im1.quantize(Q)
	# T = 3
	# np_array = np.array(im1)
	# print(np_array.shape)
	# smoothed_img = apply_gaussian_smoothing(np.array(im1),T)
	# img_laplacian_filter_1 = apply_laplacian_filter_2(Image.fromarray(smoothed_img)) 
	# print(img_laplacian_filter_1.size)
	# img_laplacian_filter_2 = apply_laplacian_filter_2(Image.fromarray(smoothed_img))
	# concatenated_feature = np.concatenate((img_laplacian_filter_1,img_laplacian_filter_2),axis = 0) 
	# print(concatenated_feature.shape)
	images = 'images/'
	save_dir = 'blob_json_LOG/'
	imgs = os.listdir(images)
	total_files = len(imgs)
	# T = [2,4,10]
	for i,img in tqdm(enumerate(imgs)):
		file_name = images + img
		fn_no_ext = img.split('.')[0] 
		# print('{}/{} Done\r'.format(i+1,total_files))
		im = Image.open(file_name)
		im = im.resize((64,64))
		im = im.quantize(Q) 
		sigma_array = [9, 15, 21, 27, 15, 27, 39, 51, 27, 51, 75, 99, 51, 99, 147, 195]
		blobs = get_all_blobs(im, sigma_array)
		blobs = blobs.tolist() 

		# print(blobs)
		# get_blobs_at_T(im, 30)
		blob_dict[fn_no_ext] = blobs

	dump_json(blob_dict,save_dir+'blob_dict.json')




		# log_feature = get_LOG_features(im, T) 
		# np.save(save_dir+fn_no_ext, log_feature)





