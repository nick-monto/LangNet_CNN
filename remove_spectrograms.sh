#!/bin/bash
echo Removing spectrograms, cause you fucked up...
for d in ./Input_spectrogram/Training/*; do
  echo Changing to $d;
  cd $d;
  for i in *.jpeg; do
    rm "$i";
    done
  cd ../../..
  done
