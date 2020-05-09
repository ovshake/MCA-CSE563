import scipy.io.wavfile as wav
from scipy import interpolate
from sys import exit 
import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt 
import os
#help taken from these resources: 
# https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520
# https://stackoverflow.com/questions/33682490/how-to-read-a-wav-file-using-scipy-at-a-different-sampling-rate
def convert_to_new_sample_rate(samples, samplerate, new_samplerate):
	time_duration = samples.shape[0] // samplerate
	old = np.linspace(0,time_duration, samples.shape[0]) 
	new = np.linspace(0,time_duration,int(samples.shape[0] * new_samplerate / samplerate))
	interpolator = interpolate.interp1d(old, samples.T)
	new_audio = interpolator(new).T
	return new_audio

def get_dft_for_signal(x):
	# print('x shape',x.shape)
	N = x.shape[0]
	# print(N) 
	# total_freqs = np.asarray([i for i in range(N)]) #N//2 + 1
	total_freq = N // 2
	intensity_at_f = np.zeros(total_freq, dtype=complex) 
	for k in range(total_freq):
		for i in range(N):
			intensity_at_f[k] = x[i] * np.exp((-2j * np.pi * i * k) / N) / N

	return intensity_at_f

def get_all_dft(X):
	spec_data = None
	for x in X:
		if spec_data is None:
			spec_data = get_dft(x).reshape(1,-1)
		else:
			spec_data = np.append(spec_data, get_dft(x).reshape(1,-1),axis = 0)
			# print(spec_data)
		# print('spec_data shape',spec_data.shape)

	return spec_data 


def fast_dft(DFT_MATRIX, signal):
	return np.dot(DFT_MATRIX, signal)

def spectrogram(sample, samplerate, plotpath = None,window = 20, overlap = .5):
	# print(samplerate)
	const_size = 16000
	if len(sample) < const_size:
		sample = np.pad(sample, (0,const_size - len(sample)), mode = 'constant')
	elif len(sample) > const_size:
		sample = sample[:const_size]

	stride_size = int((1 - overlap) * window * .001 * samplerate)
	window_size = int(window * .001 * samplerate)
	extra_part = (len(samples) - window_size) % stride_size
	# print(extra_part) 
	trimmed_samples = sample[:len(sample) -extra_part]
	# print(len(trimmed_samples)) 
	num_windows = (len(trimmed_samples) - window_size) // stride_size + 1
	# print(num_windows)
	nshape = (window_size, num_windows) 

	# print(nshape, 'nshape')
	nstrides = (trimmed_samples.strides[0], trimmed_samples.strides[0] * stride_size) 
	windows = np.lib.stride_tricks.as_strided(trimmed_samples, shape = nshape, strides=nstrides)
	# print(windows.shape)
	# print(windows)
	windows = windows.T  
	dft = []
	freqs = np.asarray([i for i in range(window_size)])
	for window in windows:
		# print(window.shape)
		dft_window = []
		for n in range(window_size // 2):
			coefficient = np.sum(window * np.exp((2j * np.pi * freqs * n) / window_size)) / window_size
			dft_window.append(np.abs(coefficient) * 2)  
		dft.append(dft_window) 
	
	# DFT_MATRIX = np.zeros((num_windows, num_windows),dtype=complex) 
	# omega = np.exp(-2j * np.pi / num_windows)
	# for i in range(num_windows):
	# 	for j in range(num_windows):
	# 		DFT_MATRIX[i][j] = omega ** (i * j) 

	# dft = fast_dft(DFT_MATRIX, windows) / (num_windows ** .5)
	dft = np.asarray(dft) 
	dft = np.transpose(dft).astype(float)
	log_spec = 10 * np.log10(dft[:-1] + 1e-8)
	plt.imshow(dft, origin='lower')
	plt.ylabel("Frequency")
	plt.xlabel("Time")
	plt.title("Spectrogram")
	plt.savefig(plotpath)
	# plt.show()

	return dft

if __name__ == '__main__':
	# audio_file = 'training/eight/004ae714_nohash_0.wav' 
	feature_folder_train = 'spectogram_features_train/'
	training_folder = 'training/'
	# validation_folder = 'validation/'
	# feature_folder_validation = 'spectogram_features_validation/'
	try:
		# os.mkdir(feature_folder_validation)
		os.mkdir(feature_folder_train)
	except:
		pass 


	# digits = os.listdir(training_folder)
	# digits = os.listdir(validation_folder)  
	digits = os.listdir(training_folder)  
	for digit in digits:
		# try:
		# 	# os.mkdir(feature_folder_validation+digit)
		# 	os.mkdir(feature_folder_train+digit)
		# 	# os.mkdir(plot_folder+digit)
		# except:
		# 	pass 
		# files = os.listdir(validation_folder + digit) 
		files = os.listdir(training_folder + digit) 
		for audio in tqdm(files):
			audio_name = audio.split('.')[0]
			# path = validation_folder + digit + '/'+audio 
			path = training_folder + digit + '/'+audio 
			samplerate, samples = wav.read(path) 
			specto = spectrogram(samples, samplerate, plotpath = digit + '_spectogram.jpeg') #plotpath=  plot_folder + digit + '/' + audio + '.jpeg'
			np.save(feature_folder_validation + digit + '/'+audio_name, specto) 
			np.save(feature_folder_train + digit + '/'+audio_name, specto) 
			
			
 
