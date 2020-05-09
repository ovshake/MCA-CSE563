import numpy as np 
from PIL import Image 
import PIL 
import os 
from tqdm import tqdm

def invalidPixel(x,y,X,Y):
	return x < 0 or x >= X-1 or y < 0 or y >= Y-1



def get_neighbours(img, x,y,distance):
	neighbours = []
	X = img.shape[0]
	Y = img.shape[1]
	for nX in range(-distance,distance+1):
		neighbours.append((x+nX,y+distance))
		neighbours.append((x+nX,y-distance))

	for nY in range(-distance,distance+1):
		neighbours.append((x+distance,y+nY))
		neighbours.append((x+distance,y-nY))


	valid_neighbours = [n for n in neighbours if not invalidPixel(n[0],n[1],X,Y)] 

	return valid_neighbours

def correlogram_at_k(img,k,Q):
	correlogram = np.zeros(Q)
	color_array = np.array(img)
	for i,x in enumerate(color_array):
		for j,y in enumerate(x):
			neighbours = get_neighbours(color_array,i,j,k) 
			# print(neighbours)
			for n in neighbours:
				if color_array[n[0]][n[1]] == y:
					correlogram[y] += 1

	correlogram /= 8*k* (color_array.shape[0] * color_array.shape[1]) 
	return correlogram 

			
def correlogram_feature(img,K,Q):
	feature_vector = np.asarray([]) 
	for k in K:
		feature_vector = np.concatenate((feature_vector,correlogram_at_k(img,k,Q)),axis=0)
	return feature_vector 
if __name__ == '__main__':
	Q = 64
	# file_name = 'all_souls_000000.jpg'
	# im1 = Image.open(file_name)  
	# im1 = im1.quantize(Q)  
	# arr = np.array(im1) 
	# print(arr.shape)
	K = [2,5,10]
	# correlogram = correlogram_at_k(im1,k,Q)
	# cor_feature = correlogram_feature(im1,K,Q)
	# print(cor_feature.shape)

	images = 'images/'
	save_dir = 'features_correlogram_without_Q/'
	imgs = os.listdir(images)
	total_files = len(imgs)
	# print(imgs)
	for i,img in tqdm(enumerate(imgs)):
		file_name = images + img
		fn_no_ext = img.split('.')[0] 
		# print('{}/{} Done\r'.format(i+1,total_files))
		im = Image.open(file_name)
		im = im.resize((128,128))
		im = im.quantize(Q) 
		cor_feature = correlogram_feature(im, K, Q) 
		np.save(save_dir+fn_no_ext, cor_feature)















