#!/usr/bin/env python
import os
import numpy as np
from my_spectrogram import my_specgram
from collections import OrderedDict
from scipy.io import wavfile
import matplotlib.pylab as plt

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_audio_wav/'
OUTPUT_FOLDER = 'Input_spectrogram/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

audio_dict = OrderedDict()

for l in languages:
    audio_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))


def plot_spectrogram(audiopath, plotpath=None, NFFT_window=0.025,
                     noverlap_window=0.022, freq_min=None, freq_max=None,
                     axis='on'):
    fs, data = wavfile.read(audiopath)
    NFFT = int(fs*NFFT_window)  # 50ms window
    noverlap = int(fs*noverlap_window)
    fc = int(np.sqrt(freq_min*freq_max))
    # Pxx is the segments x freqs array of instantaneous power, freqs is
    # the frequency vector, bins are the centers of the time bins in which
    # the power is computed, and im is the matplotlib.image.AxesImage
    # instance
    Pxx, freqs, bins, im = my_specgram(data, NFFT=NFFT, Fs=fs,
                                       Fc=fc, detrend=None,
                                       window=np.hanning(NFFT),
                                       noverlap=noverlap, cmap='Greys',
                                       xextent=None,
                                       pad_to=None, sides='default',
                                       scale_by_freq=None,
                                       minfreq=freq_min, maxfreq=freq_max)
    plt.axis(axis)
    if plotpath:
        plt.savefig(plotpath, bbox_inches='tight', transparent=False)
    else:
        plt.show()

    plt.clf()

# Checking out some other ways to plot spectrograms
# from matplotlib.mlab import specgram
# from scipy.signal import gaussian
# fs, data = wavfile.read('Input_audio_wav/Italian/ita-0011acd6.wav')
# Pxx, freqs, bins, im = plt.specgram(data, NFFT=2048, Fs=fs, detrend=None,
#                                     window=gaussian(2048, 128, sym=False),
#                                     noverlap=1800, pad_to=3000)
#
# plt.ylim(0, 5500)
# plt.show()


# test function
# plot_spectrogram('Input_audio_wav/Italian/ita-0011acd6.wav',
#                  NFFT_window=0.025, noverlap_window=0.023,
#                  freq_min=30, freq_max=5500)

# create spectrograms of randomly drawn samples from each language
import fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result[0]


random_wav = []
for key in audio_dict:
    random_wav.append(sorted(np.random.choice(audio_dict[key], 100, replace=False)))

if not os.path.exists(OUTPUT_FOLDER + 'Training'):
    os.makedirs(OUTPUT_FOLDER + 'Training')
    print('Successfully created a training folder!')
print('Populating training folder with spectrograms...')
for i in range(0, len(random_wav)):
    if not os.path.exists(OUTPUT_FOLDER + 'Training/' + str(languages[i])):
        os.makedirs(OUTPUT_FOLDER + 'Training/' + str(languages[i]))
        print('Successfully created a {} training folder!'.format(languages[i]))
    print('Populating {} training folder with spectrograms...'.format(languages[i]))
    for j in range(0, len(random_wav[i])):
        for k in range(0, 10):
            plot_spectrogram(find(random_wav[i][j], INPUT_FOLDER),
                             plotpath=OUTPUT_FOLDER + 'Training/' +
                                      str(languages[i]) + '/' +
                                      str(random_wav[i][j][:-4]) + '_' +
                                      str(k) + '.jpeg',
                             NFFT_window=0.025, noverlap_window=0.023,
                             freq_min=30, freq_max=5500)
        print('Done with {}.'.format(random_wav[i][j][:-4]))


# # spectrogram loop for the first 100 audio files in each language
# for key in audio_dict:
#     if not os.path.exists(OUTPUT_FOLDER + str(key)):
#         os.makedirs(OUTPUT_FOLDER + str(key))
#         print('Successfully created a folder for ' + str(key) + '!')
#     print('Moving to the {} folder.'.format(key))
#     os.chdir(OUTPUT_FOLDER + str(key))
#     print('Populating with spectrograms...')
#     for i in range(0, 100):  # create spectrograms for the first 100 audiofiles
#         for j in range(0, 10):
#             plot_spectrogram('../../' + INPUT_FOLDER + str(key) + '/' +
#                              audio_dict[key][i],
#                              plotpath=str(audio_dict[key][i][:-4]) + '_' + str(j) + '.jpeg',
#                              NFFT_window=0.05, noverlap_window=0.045,
#                              freq_min=30, freq_max=5500)
#         print('Done with {}. Moving onto {}.'.format(audio_dict[key][i][:-4],
#                                                      audio_dict[key][i+1][:-4]))
#     print('Going back to original directory.')
#     os.chdir(SCRIPT_DIR)
