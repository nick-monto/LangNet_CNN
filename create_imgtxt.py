import os

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_spectrogram/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

spec_dict = {}

for l in languages:
    spec_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))

txtfile = open('./img_set.txt', mode='w')

for key in spec_dict:
    os.chdir(INPUT_FOLDER + str(key))
    for i in range(0, 100):
        txtfile.write(str(spec_dict[key][i]) + " " + str(key))
        txtfile.write("\n")
    os.chdir(SCRIPT_DIR)

txtfile.close()

# this will be used for setting up the stim labels
import pandas as pd
stim = pd.read_table('img_set.txt', delim_whitespace=True, header=None)
