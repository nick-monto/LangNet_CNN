#!/usr/bin/env python
from pydub import AudioSegment
import os
import subprocess


SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = './Input_audio/'
OUTPUT_FOLDER = './Input_audio_wav/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

audio_dict = {}

for i in languages and s in os.listdir(INPUT_FOLDER + languages + '/'):
    audio_dict[i] = audiotools.open(INPUT_FOLDER + i + '/' + s)

audio_list = []

for i in languages:
    for f in os.listdir(INPUT_FOLDER + i + '/'):
        if f.endswith('.flac'):
            audio_list.append(f)
for i in languages:
    os.makedirs(OUTPUT_FOLDER + str(i))
    print('Successfully created a folder for ' + str(i) + '!')

# was hoping that this would convert the Flacs to Wav
# ...but it only copied the files
for i in languages:
    for f in os.listdir(INPUT_FOLDER + i + '/'):
        if f.endswith('.flac'):
            AudioSegment.from_file(INPUT_FOLDER + i + '/' +
                                   str(f)).export(OUTPUT_FOLDER + i +
                                                  '/' + str(f), format="wav")
