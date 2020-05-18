from sklearn.manifold import TSNE
import numpy as np
from Skipgram import * 
from dataloader import * 
import os 
import torch
import torch as ch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import random
random.seed(0)

def get_word_vec(word, WORD_TO_IDX, vocab_size):
	input = torch.zeros(vocab_size)
	input[WORD_TO_IDX[word]] = 1
	return input 

def plot_tsne(words_vec, epoch_num):
	embed_vec = []
	word_array = []
	for k in words_vec:
		embed_vec.append(words_vec[k])
		word_array.append(k) 
	embed_vec = np.asarray(embed_vec) 
	X_embedded = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=12).fit_transform(embed_vec)
	colors = cm.rainbow(np.linspace(0, 1, len(words_vec)))
	plt.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5)
	for i,(x,y) in enumerate(X_embedded):
		print(word_array[i], x, y)
		plt.annotate(word_array[i], (x, y)) 

	plt.savefig('Epoch Num: {}.jpeg'.format(epoch_num))




	

ckpts = [25,50,75,100,150,200]
embed_size = 300
abc_corpus = ABCCorpus() 
models = [SkipGram(abc_corpus.vocab_size, embed_size) for i in ckpts] 
for i,model in enumerate(models):
	model.load_state_dict(torch.load('word2vec_epoch_{}_ctx_2.pth'.format(ckpts[i]))) 

vocab_words = abc_corpus.WORD_TO_IDX.keys()
words = random.sample(vocab_words, 20)
 

for i,model in enumerate(models):
	words_vec = {}
	for word in words:
		vec = get_word_vec(word, abc_corpus.WORD_TO_IDX, abc_corpus.vocab_size) 
		words_vec[word] = model.embedding_layer(vec).detach().numpy() 
	plot_tsne(words_vec, ckpts[i])

