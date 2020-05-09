from sklearn.svm import SVC 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score , classification_report
from tqdm import tqdm
import os
import pickle 
from sys import exit 
from sklearn.preprocessing import MinMaxScaler
import pickle

if __name__ == '__main__':
	word_nums = ['zero','one','two','three','four','five','six','seven','eight','nine']
	word_to_num = {}
	num_to_word = {} 
	for i in range(10):
		word_to_num[word_nums[i]] = i 
		num_to_word[i] = word_nums[i]

	mfcc_train = 'mfcc_features_train/'
	spectogram_train = 'spectogram_features_train/' 
	mfcc_validation = 'mfcc_features_validation/'
	spectogram_validation = 'spectogram_features_validation/'
	X_train_spec = [] 
	y_train_spec = [] 
	X_test_spec = []
	y_test_spec = [] 
	X_train_mfcc = [] 
	y_train_mfcc = [] 
	X_test_mfcc = []
	y_test_mfcc = [] 
	for num in word_nums:
		
		pth = mfcc_train + num + '/'
		files = os.listdir(pth) 
		for file in tqdm(files):
			x = np.load(pth + file)
			x = x.flatten() 
			# print(x.shape, 'mfcc')
			if x.shape[0] < 1188:
				x = np.pad(x, (0,1188 - x.shape[0]), mode = 'constant')
			elif x.shape[0] > 1188:
				x = x[:1188]
			if x.shape[0] != 1188:
				exit()
			X_train_mfcc.append(x)
			y_train_mfcc.append(word_to_num[num])

		pth = spectogram_train + num + '/'
		files = os.listdir(pth) 
		for file in tqdm(files):
			x = np.load(pth + file)
			x = x.flatten()
			# print(x.shape, 'spec') 
			if x.shape[0] < 15840:
				x = np.pad(x, (0,15840 - x.shape[0]), mode = 'constant')
			elif x.shape[0] > 15840:
				x = x[:15840]
			if x.shape[0] != 15840:
				exit()
			X_train_spec.append(x)
			y_train_spec.append(word_to_num[num])

		pth = mfcc_validation + num + '/'
		files = os.listdir(pth) 
		for file in tqdm(files):
			x = np.load(pth + file)
			x = x.flatten() 
			# print(x.shape, 'mfcc')
			if x.shape[0] < 1188:
				x = np.pad(x, (0,1188 - x.shape[0]), mode = 'constant')
			elif x.shape[0] > 1188:
				x = x[:1188]
			if x.shape[0] != 1188:
				exit()
			X_test_mfcc.append(x)
			y_test_mfcc.append(word_to_num[num])

		pth = spectogram_validation + num + '/' 
		files = os.listdir(pth) 
		for file in tqdm(files):
			x = np.load(pth + file)
			x = x.flatten() 
			# print(x.shape,'spec')
			if x.shape[0] < 15840:
				x = np.pad(x, (0,15840 - x.shape[0]), mode = 'constant')
			elif x.shape[0] > 15840:
				x = x[:15840]
			if x.shape[0] != 15840:
				exit()
			X_test_spec.append(x)
			y_test_spec.append(word_to_num[num])


	np.save('X_train_mfcc',X_train_mfcc) 
	np.save('y_train_mfcc', y_train_mfcc)
	np.save('X_test_mfcc',X_test_mfcc) 
	np.save('y_test_mfcc', y_test_mfcc)
	np.save('X_train_spec', X_train_spec) 
	np.save('y_train_spec',y_train_spec)
	np.save('X_test_spec', X_test_spec) 
	np.save('y_test_spec',y_test_spec)



	X_train_mfcc = np.load('X_train_mfcc.npy') 
	y_train_mfcc = np.load('y_train_mfcc.npy')
	y_test_mfcc = np.load('y_test_mfcc.npy')
	X_test_mfcc = np.load('X_test_mfcc.npy') 
	X_train_spec = np.load('X_train_spec.npy') 
	y_train_spec = np.load('y_train_spec.npy')
	X_test_spec = np.load('X_test_spec.npy') 
	y_test_spec = np.load('y_test_spec.npy')
	
	# scaling_mfcc = MinMaxScaler(feature_range=(-1,1)).fit(X_train_mfcc)
	scaling_spec = MinMaxScaler(feature_range=(-1,1)).fit(X_train_spec)
	
	# X_train_mfcc = scaling_mfcc.transform(X_train_mfcc) 
	# X_test_mfcc = scaling_mfcc.transform(X_test_mfcc)

	X_train_spec = scaling_spec.transform(X_train_spec)
	X_test_spec = scaling_spec.transform(X_test_spec)

	clf_spec = SVC(verbose=True,kernel = 'linear')
	# clf_mfcc = SVC(verbose=True,  kernel = 'linear') 


	# clf_mfcc.fit(X_train_mfcc, y_train_mfcc) 
	# mfcc_y_pred = clf_mfcc.predict(X_test_mfcc) 
	clf_spec.fit(X_train_spec, y_train_spec) 
	spec_y_pred = clf_spec.predict(X_test_spec) 
	


	# print('Accuracy Score for MFCC: {}'.format(accuracy_score(mfcc_y_pred, y_test_mfcc)))
	# print('classification_report for MFCC')
	# print(classification_report(y_test_mfcc, mfcc_y_pred))
	# clf_spec = pickle.load(open('model_spec.pth','rb'))
	print('Accuracy Score for Spectogram: {}'.format(accuracy_score(spec_y_pred, y_test_spec)))
	pickle.dump(clf_spec, open('model_spec.pth','wb'))
	print('classification_report for Specotgram')
	print(classification_report(y_test_spec, spec_y_pred))







