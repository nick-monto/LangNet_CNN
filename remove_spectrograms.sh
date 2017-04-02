#!/bin/bash
for d in ./Input_spectrogram/*; do
  echo Changing to $d;
  cd $d;
  echo Removing spectrograms, cause you fucked up...
  for i in *.png; do
    rm "$i";
    done
  echo Changing back to original directory;
  cd ../..
  done
