from torch.utils.data import Dataset, DataLoader
import os
import nltk 
import warnings
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
import torch
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words('english')) 
ps = PorterStemmer() 

class ABCCorpus(Dataset):
	def __init__(self):
		self.word, self.ctx, self.WORD_TO_IDX = self.generate_ctx_pairs_abc() 
		self.vocab_size = len(self.WORD_TO_IDX)

	def __len__(self):
		return len(self.word) 

	def __getitem__(self, idx):
		word_item = self.word[idx] 
		ctx_item = self.ctx[idx] 
		input = torch.zeros(self.vocab_size) 
		input[self.WORD_TO_IDX[word_item]] = 1 
		return input, self.WORD_TO_IDX[ctx_item] 


	def generate_ctx_pairs_abc(self, context = 2):
		abc_sents = nltk.corpus.abc.sents()
		word = []
		ctx = [] 
		WORD_TO_IDX = {} 
		for sent in abc_sents:
			for idx, wrd in enumerate(sent):
				if wrd in stop_words:
					continue
				stemmed_word = ps.stem(wrd)
				if stemmed_word not in WORD_TO_IDX:
					WORD_TO_IDX[stemmed_word] = len(WORD_TO_IDX)

				left_border = max(0, idx - context)
				right_border = min(len(sent), idx + context)
				for ctx_idx in range(left_border, right_border):
					if ctx_idx == idx or sent[ctx_idx] in stop_words:
						continue
					if sent[ctx_idx] not in WORD_TO_IDX:
						WORD_TO_IDX[sent[ctx_idx]] = len(WORD_TO_IDX)

					word.append(stemmed_word)
					ctx.append(sent[ctx_idx]) 

		return word, ctx, WORD_TO_IDX 






def test():
	abc_dataset = ABCCorpus() 
	return abc_dataset 


if __name__ == '__main__':
	print(len(test()) )

		


