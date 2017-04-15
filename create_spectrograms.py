import os
import numpy as np
from my_spectrogram import my_specgram
from collections import OrderedDict
from scipy.io import wavfile
import matplotlib.pylab as plt
from itertools import repeat


SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_audio_wav/'
OUTPUT_FOLDER = 'Input_spectrogram/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

audio_dict = OrderedDict()

for l in languages:
    audio_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))


def plot_spectrogram(audiopath, plotpath=None, NFFT_window=0.05,
                     noverlap_window=0.045, freq_min=None, freq_max=None,
                     axis='on'):
    fs, data = wavfile.read(audiopath)
    NFFT = int(fs*NFFT_window)  # 50ms window
    noverlap = int(fs*noverlap_window)
    fc = int((freq_min+freq_max)/2)
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
        plt.savefig(plotpath, bbox_inches='tight', transparent=True)
    else:
        plt.show()

    plt.clf()


# test function
# plot_spectrogram('Input_audio_wav/Italian/ita-0011acd6.wav', plotpath=None,
#                  NFFT_window=0.05, noverlap_window=0.045,
#                  freq_min=30, freq_max=5500)

# constructing spectrogram loop
for key in audio_dict:
    if not os.path.exists(OUTPUT_FOLDER + str(key)):
        os.makedirs(OUTPUT_FOLDER + str(key))
        print('Successfully created a folder for ' + str(key) + '!')
    print('Moving to the {} folder.'.format(key))
    os.chdir(OUTPUT_FOLDER + str(key))
    print('Populating with spectrograms...')
    for i in range(0, 100):  # create spectrograms for the first 100 audiofiles
        for j in range(0, 20):
            plot_spectrogram('../../' + INPUT_FOLDER + str(key) + '/' +
                             audio_dict[key][i],
                             plotpath=str(audio_dict[key][i][:-4]) + '_' + str(j) + '.png',
                             NFFT_window=0.05, noverlap_window=0.045,
                             freq_min=30, freq_max=5500)
        print('Done with {}. Moving onto {}.'.format(audio_dict[key][i][:-4],
                                                     audio_dict[key][i+1][:-4]))
    print('Going back to original directory.')
    os.chdir(SCRIPT_DIR)
