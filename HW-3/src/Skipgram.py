import torch.nn as nn

class SkipGram(nn.Module):
	def __init__(self, vocab_size, emb_dim):
		super(SkipGram, self).__init__() 
		self.vocab_size = vocab_size 
		self.emb_dim = emb_dim 
		self.embedding_layer = nn.Linear(self.vocab_size, self.emb_dim) 
		self.output_layer = nn.Linear(self.emb_dim, self.vocab_size) 

	def forward(self, x):
		return self.output_layer(self.embedding_layer(x)) 




