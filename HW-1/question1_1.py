import numpy as np 
from PIL import Image 
import PIL 
import os 

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


def get_query(filename):
	f = open(filename,'r')
	line = f.readline() 
	query = line.split()[0] 
	query = '_'.join(query.split('_')[1:])
	return query

def get_image_feature(img):
	Q= 64
	images = 'images/'
	file_name = images + img
	# print('{}/{} Done\r'.format(i+1,total_files))
	im = Image.open(file_name)
	K = [2,5,10]
	im = im.resize((128,128))
	im = im.quantize(Q) 
	cor_feature = correlogram_feature(im, K, Q) 
	return cor_feature 


def get_similarity(feature1, feature2):
	feature1 = feature1.reshape((3,-1))
	feature2 = feature2.reshape((3,-1))
	distance = 0
	for c in range(feature1.shape[0]):
		for k in range(feature1.shape[1]):
			distance += (np.abs(feature1[c][k] - feature2[c][k]) / (1 + feature1[c][k] + feature2[c][k]))

	return abs(distance)


def get_top_K_results(qeury_feature, K):
	feature_folder = 'features_correlogram/'
	all_img = os.listdir(feature_folder)
	results = [] 
	for img in all_img:
		feature = np.load(feature_folder + img) 
		sim = get_similarity(feature, qeury_feature) 
		results.append( (sim, img.split('.')[0] ))

	results.sort(reverse=True) 
	results = [i[1] for i in results]
	return results[:K] 

def get_ground_truth(query_name):
	ground_truth = 'train/ground_truth/'
	types = ['good','ok','junk']
	truths = []
	for t in types:
		f = open(ground_truth + query_name +'_' +t + '.txt','r') 
		lines = f.readlines()
		lines = [(line.strip(),t) for line in lines]
		truths.extend(lines) 

	return truths 

if __name__ == '__main__':
	query_folder = 'train/query/'
	ground_truth = 'train/ground_truth/'
	image_folder = 'images/'
	queries = os.listdir(query_folder)
	K = 500
	report = 'report2.txt'
	report = open(report,'w')
	for query in queries:
		query_name_offset = query.find('_query.txt') 
		query_name = query[:query_name_offset] 
		print(query_name)
		query_img = get_query(query_folder+query)
		query_feature = get_image_feature(query_img + '.jpg')
		results = get_top_K_results(query_feature, K) 
		all_truths = get_ground_truth(query_name)
		good = [i[0] for i in all_truths if i[1] == 'good']
		ok = [i[0] for i in all_truths if i[1] == 'ok']
		junk  = [i[0] for i in all_truths if i[1] == 'junk']
		truths = good + ok + junk
		precision_array = [] 
		recall_array = []
		f1_array = []
		for k in [300,350,400,450,500]:
			res = results[:k]
			matches = set(truths) & set(res)
			precision = len(matches) / k 
			recall = len(matches) / len(truths) 
			f1_score = (2*precision * recall) / (precision + recall) if precision != 0 or recall != 0 else 0
			num_good = len(set(good) & set(res))
			num_ok = len(set(ok) & set(res))
			num_junk = len(set(junk) & set(res))
			precision_array.append(precision) 
			recall_array.append(recall) 
			f1_array.append(f1_score) 
			p = '{}, Precision At {}: {}'.format(query, k , precision)
			r = '{}, Recall At {}: {}'.format(query, k , recall)
			f = '{}, F1 At {}: {}'.format(query, k , f1_score)
			g = 'Good at {}: {}'.format(k,num_good)
			o = 'Ok at {}: {}'.format(k,num_ok)
			j = 'Junk at {}: {}'.format(k,num_junk)
			print(g)
			print(o)
			print(j)
			print(p) 
			print(r)
			print(f) 
			report.write(g + '\n')
			report.write(o + '\n')
			report.write(j + '\n')
			report.write(p + '\n') 
			report.write(r + '\n') 
			report.write(f + '\n')

		min_p = min(precision_array) 
		max_p = max(precision_array) 
		avg_p = sum(precision_array) / len(precision_array) 

		min_r = min(recall_array) 
		max_r = max(recall_array) 
		avg_r = sum(recall_array) / len(recall_array) 

		min_f = min(f1_array) 
		max_f = max(f1_array) 
		avg_f = sum(f1_array) / len(f1_array) 

		min_p_str = 'Mean Precision is: {}'.format(min_p)
		max_p_str = 'Mean Precision is: {}'.format(max_p)
		avg_p_str = 'Mean Precision is: {}'.format(avg_p)

		min_r_str = 'Mean Precision is: {}'.format(min_r)
		max_r_str = 'Mean Precision is: {}'.format(max_r)
		avg_r_str = 'Mean Precision is: {}'.format(avg_r)

		min_f_str = 'Mean Precision is: {}'.format(min_f)
		max_f_str = 'Mean Precision is: {}'.format(max_f)
		avg_f_str = 'Mean Precision is: {}'.format(avg_f)

		write_stuff = [min_p_str, max_p_str, avg_p_str, min_r_str, max_r_str, avg_r_str, min_f_str, max_f_str, avg_f_str]
		for w in write_stuff:
			report.write(w + '\n')
	
	report.close()














