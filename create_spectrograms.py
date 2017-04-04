import os
import numpy as np
from collections import OrderedDict
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from pylab import *
from matplotlib import *

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_audio_wav/'
OUTPUT_FOLDER = 'Input_spectrogram/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

audio_dict = OrderedDict()

for l in languages:
    audio_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))


def my_specgram(x, NFFT=256, Fs=2, Fc=0, detrend=None,
                window=mlab.window_hanning, noverlap=128,
                cmap=None, xextent=None, pad_to=None, sides='default',
                scale_by_freq=None, minfreq=None, maxfreq=None, **kwargs):
    """
    Credit: http://stackoverflow.com/questions/19468923/cutting-of-unused-frequencies-in-specgram-matplotlib --user: wwii

    call signature::

      specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=128,
               cmap=None, xextent=None, pad_to=None, sides='default',
               scale_by_freq=None, minfreq = None, maxfreq = None, **kwargs)

    Compute a spectrogram of data in *x*.  Data are split into
    *NFFT* length segments and the PSD of each section is
    computed.  The windowing function *window* is applied to each
    segment, and the amount of overlap of each segment is
    specified with *noverlap*.

    %(PSD)s

      *Fc*: integer
        The center frequency of *x* (defaults to 0), which offsets
        the y extents of the plot to reflect the frequency range used
        when a signal is acquired and then filtered and downsampled to
        baseband.

      *cmap*:
        A :class:`matplotlib.cm.Colormap` instance; if *None* use
        default determined by rc

      *xextent*:
        The image extent along the x-axis. xextent = (xmin,xmax)
        The default is (0,max(bins)), where bins is the return
        value from :func:`mlab.specgram`

      *minfreq, maxfreq*
        Limits y-axis. Both required

      *kwargs*:

        Additional kwargs are passed on to imshow which makes the
        specgram image

      Return value is (*Pxx*, *freqs*, *bins*, *im*):

      - *bins* are the time points the spectrogram is calculated over
      - *freqs* is an array of frequencies
      - *Pxx* is a len(times) x len(freqs) array of power
      - *im* is a :class:`matplotlib.image.AxesImage` instance

    Note: If *x* is real (i.e. non-complex), only the positive
    spectrum is shown.  If *x* is complex, both positive and
    negative parts of the spectrum are shown.  This can be
    overridden using the *sides* keyword argument.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/specgram_demo.py

    """

    #####################################
    # modified  axes.specgram() to limit
    # the frequencies plotted
    #####################################

    # this will fail if there isn't a current axis in the global scope
    ax = gca()
    Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
                                     window, noverlap, pad_to,
                                     sides, scale_by_freq)

    # modified here
    #####################################
    if minfreq is not None and maxfreq is not None:
        Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
        freqs = freqs[(freqs >= minfreq) & (freqs <= maxfreq)]
    #####################################

    Z = 10. * np.log10(Pxx)
    Z = np.flipud(Z)

    if xextent is None: xextent = 0, np.amax(bins)
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]
    im = ax.imshow(Z, cmap, extent=extent, **kwargs)
    ax.axis('auto')

    return Pxx, freqs, bins, im


def plot_spectrogram(audiopath, plotpath=None, NFFT=1024,
                     freq_min=None, freq_max=None):
    fs, data = wavfile.read(audiopath)

    # Pxx is the segments x freqs array of instantaneous power, freqs is
    # the frequency vector, bins are the centers of the time bins in which
    # the power is computed, and im is the matplotlib.image.AxesImage
    # instance
    Pxx, freqs, bins, im = my_specgram(data, NFFT=NFFT, Fs=fs,
                                       Fc=0, detrend=None,
                                       window=np.hanning(NFFT),
                                       noverlap=512, cmap='greys',
                                       xextent=None,
                                       pad_to=None, sides='default',
                                       scale_by_freq=None,
                                       minfreq=freq_min, maxfreq=freq_max)
    plt.axis('off')
    if plotpath:
        pyplot.savefig(plotpath, bbox_inches='tight', transparent=True)
    else:
        pyplot.show()

    pyplot.clf()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def apply_bpFilter(audiopath, lowcut, highcut, order=5):
    fs, data = wavfile.read(audiopath)
    filt_data = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)
    return filt_data


# test function
plot_spectrogram(INPUT_FOLDER + languages[0], plotpath=None, NFFT=1024,
                 freq_min=0, freq_max=5500)

# constructing spectrogram loop
for key in audio_dict:
    if not os.path.exists(OUTPUT_FOLDER + str(key)):
        os.makedirs(OUTPUT_FOLDER + str(key))
    print('Successfully created a folder for ' + str(key) + '!')
    os.chdir(OUTPUT_FOLDER + str(key))
    print('Populating with spectrograms...')
    for i in range(0, 100):  # create spectrograms for the first 100 audiofiles
        plot_spectrogram('../../' + INPUT_FOLDER + str(key) + '/' +
                         audio_dict[key][i],
                         plotpath=str(audio_dict[key][i][:-4]) + '.png',
                         NFFT=1024,
                         freq_min=0, freq_max=5500)
    print('Going back to original directory.')
    os.chdir(SCRIPT_DIR)
