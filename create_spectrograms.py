import os
from collections import OrderedDict
from scipy.io import wavfile
from matplotlib import pyplot as plt


SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_audio_wav/'
OUTPUT_FOLDER = 'Input_spectrogram/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

audio_dict = OrderedDict()

for l in languages:
    audio_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))

# constructing spectrogram loop
def plot_spectrogram(audiopath, binsize=2**10, plotpath=None, ylim=(0, 5500)):
    fs, data = wavfile.read(audiopath)

    # Pxx is the segments x freqs array of instantaneous power, freqs is
    # the frequency vector, bins are the centers of the time bins in which
    # the power is computed, and im is the matplotlib.image.AxesImage
    # instance
    Pxx, freqs, bins, im = plt.specgram(data, NFFT=2**10, Fs=fs, noverlap=900)
    plt.ylim(ylim)
    plt.axis('off')
    if plotpath:
        plt.savefig(plotpath, bbox_inches='tight', transparent=True)
    else:
        plt.show()

    plt.clf()


for key in audio_dict:
    if not os.path.exists(OUTPUT_FOLDER + str(key)):
        os.makedirs(OUTPUT_FOLDER + str(key))
    print('Successfully created a folder for ' + str(key) + '!')
    os.chdir(OUTPUT_FOLDER + str(key))
    print('Populating with spectrograms...')
    for i in range(0, 100):  # create spectrograms for the first 100 audiofiles
        plot_spectrogram('../../' + INPUT_FOLDER + str(key) + '/' +
                         audio_dict[key][i],
                         binsize=2**10,
                         plotpath=str(audio_dict[key][i][:-4]) + '.png',
                         ylim=(0, 5500))
    print('Going back to original directory.')
    os.chdir(SCRIPT_DIR)
