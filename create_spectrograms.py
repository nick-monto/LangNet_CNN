#!/usr/bin/env python
import os
import numpy as np
import math
from my_spectrogram import my_specgram
from collections import OrderedDict
from scipy.io import wavfile
import matplotlib.pylab as plt
from pylab import rcParams

rcParams['figure.figsize'] = 6, 3

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_audio_wav/'
OUTPUT_FOLDER = 'Input_spectrogram/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

audio_dict = OrderedDict()

for l in languages:
    audio_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))


def plot_spectrogram(audiopath, plotpath=None, NFFT_window=0.25,
                     noverlap_window=0.022, freq_min=None, freq_max=None,
                     axis='on'):
    fs, data = wavfile.read(audiopath)
    data = data + np.random.normal(200, 75, len(data))
    NFFT = pow(2, int(math.log(int(fs*NFFT_window), 2) + 0.5))  # 25ms window, nearest power of 2
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
    plt.axis('off')
    im.axes.axis('tight')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    if plotpath:
        plt.savefig(plotpath, bbox_inches='tight',
                    transparent=False, pad_inches=0, dpi=96)
    else:
        plt.show()
    plt.clf()


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
