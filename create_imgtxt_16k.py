import os

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_spectrogram_16k/Training/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

spec_dict = {}

for l in languages:
    spec_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))

txtfile = open('./img_set_16k.txt', mode='w')

for l in languages:
    os.chdir(INPUT_FOLDER + str(l))
    for i in range(0, len(spec_dict[l])):
        txtfile.write(str(spec_dict[l][i]) + " " + str(l))
        txtfile.write("\n")
    os.chdir(SCRIPT_DIR)

txtfile.close()
