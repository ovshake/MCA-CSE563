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
from Skipgram import *
from dataloader  import *  

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



abc_dataset = ABCCorpus() 
vocab_size = abc_dataset.vocab_size
embedding_dim = 300
model = SkipGram(vocab_size, embedding_dim) 

train_loader = ch.utils.data.DataLoader(abc_dataset, batch_size = 128, num_workers = 2)
num_epochs = 200 
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.1,
					  momentum=0.9, weight_decay=5e-4)

ckpts = [25,50,75,100,150,200]
model.train() 
for iter in range(num_epochs):
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		print('Epoch: {} Batch No. {} Loss: {} Accuracy: {} Correct: {} Total: {}'\
				.format(iter, batch_idx + 1, train_loss / (batch_idx + 1), 100.* correct / total, correct,
					total), end = '\r')

	if iter + 1 in ckpts:
		print('Saving Checkpoint')
		torch.save(model.state_dict(), 'word2vec_epoch_{}_ctx_{}.pth'.format(iter+1,2))



