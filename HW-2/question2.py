import scipy.io.wavfile as wav
from scipy import interpolate
from sys import exit 
import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt 
import os 
from scipy.fftpack import dct

def get_mfcc(sample, samplerate, plotpath, window = 20, overlap = .5):
	global n_dict, lift_dict, fbank_dict
	#pre_emphasis 
	pre_emph_rate = 0.97 
	const_size = 16000
	# print(len(sample),'1')
	if len(sample) < const_size:
		# print('lkdslkdnfld')
		sample = np.pad(sample, (0,const_size - len(sample)), mode = 'constant')
	# print(len(sample),2)
	# print(len(sample), 'fdlkanflkdf')
	emph_sample = np.append(sample[0], sample[1:] - pre_emph_rate * sample[:-1]) 
	sample = emph_sample
	stride_size = int((1 - overlap) * window * .001 * samplerate)
	window_size = int(window * .001 * samplerate)
	extra_part = (len(samples) - window_size) % stride_size
	trimmed_samples = sample[:len(sample) -extra_part]
	num_windows = (len(trimmed_samples) - window_size) // stride_size + 1
	nshape = (window_size, num_windows) 
	nstrides = (trimmed_samples.strides[0], trimmed_samples.strides[0] * stride_size) 
	windows = np.lib.stride_tricks.as_strided(trimmed_samples, shape = nshape, strides=nstrides)
	# print(windows.shape)
	# print(windows)
	windows *= np.hamming(window_size)[:, None]
	windows = windows.T
	NFFT = 512
	windows = ((1/NFFT) * ((np.absolute(np.fft.rfft(windows, NFFT))) ** 2))
	nfilt = 40
	num_ceps = 12
	cep_lifter = 22
	low_freq_mel = 0
	if samplerate in fbank_dict:
		filter_banks = fbank_dict[samplerate]
	else:
		high_freq_mel = (2595 * np.log10(1 + (samplerate / 2) / 700))  # Convert Hz to Mel
		mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
		hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
		bin = np.floor((NFFT + 1) * hz_points / samplerate)
		fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
		for m in range(1, nfilt + 1):
			f_m_minus = int(bin[m - 1])   # left
			f_m = int(bin[m])             # center
			f_m_plus = int(bin[m + 1])    # right
			for k in range(f_m_minus, f_m):
				fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
			for k in range(f_m, f_m_plus):
				fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
		filter_banks = np.dot(windows, fbank.T)
		filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
		filter_banks = 20 * np.log10(filter_banks)  # dB
		fbank_dict[samplerate] = filter_banks

	filter_banks = fbank_dict[samplerate]
	mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
	(nframes, ncoeff) = mfcc.shape
	# print(mfcc.shape)
	# n = np.arange(ncoeff)
	# if ncoeff not in n_dict:
	# 	print('New Key Added: {}'.format(ncoeff))
	# 	n_dict[ncoeff] = np.arange(ncoeff)
	# n = n_dict[ncoeff]
	if ncoeff not in lift_dict:
		print('New Key Added: {}'.format(ncoeff))
		lift_dict[ncoeff] = 1 + (cep_lifter / 2) * np.sin(np.pi * np.arange(ncoeff) / cep_lifter)
	lift = lift_dict[ncoeff]
	mfcc *= lift
	mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
	plt.imshow(mfcc.T)
	# plt.show()
	plt.savefig(plotpath)
	return mfcc 

	# dft = []
	# for window in windows:
	# 	# print(window.shape)
	# 	dft_window = []
	# 	for n in range(window_size // 2):
	# 		freqs = np.asarray([i for i in range(window_size)])
	# 		coefficient = np.sum(window * np.exp((2j * np.pi * freqs * n) / window_size)) / window_size
	# 		dft_window.append(np.abs(coefficient) * 2)  
	# 	dft.append(dft_window) 

	# dft = np.asarray(dft) 
	# dft = np.transpose(dft).astype(float)
	# log_spec = 10 * np.log10(dft[:-1])
	# plt.imshow(dft, origin='lower')
	# plt.ylabel("Frequency")
	# plt.xlabel("Time")
	# plt.title("Spectrogram")
	# plt.savefig(plotpath)
	# plt.show()




if __name__ == '__main__':
	# audio_file = 'training/eight/004ae714_nohash_0.wav' 
	# samplerate, samples = wav.read(audio_file)

	# spec = get_mfcc(samples, samplerate, 'flksjdf')

	n_dict = {}
	lift_dict = {}
	fbank_dict = {}


	# feature_folder_train = 'mfcc_features_train/'
	plot_folder = 'mfcc_plots/'
	# training_folder = 'training/'
	validation_folder = 'validation/'
	# feature_folder_validation = 'mfcc_features_validation/'
	digits = os.listdir(validation_folder) 
	# try:
	# 	os.mkdir(feature_folder_validation)
	# except:
	# 	pass 

	for digit in digits:
		try:
			os.mkdir(feature_folder_validation+digit)
			# os.mkdir(plot_folder+digit)
		except:
			pass 
		files = os.listdir(validation_folder + digit) 
		for audio in tqdm(files):
			audio_name = audio.split('.')[0]
			path = validation_folder + digit + '/'+audio 
			samplerate, samples = wav.read(path) 
			mfcc = get_mfcc(samples, samplerate, digit + '_mfcc.jpeg') #plot_folder + digit + '/' + audio + '.jpeg'
			# np.save(feature_folder_validation + digit + '/'+audio_name, mfcc)
			break 



#please note help taken from: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html for implementation details.