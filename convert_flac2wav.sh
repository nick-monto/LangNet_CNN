#!/bin/bash
for d in ./Input_audio_wav/*; do
  echo Changing to $d;
  cd $d;
  echo Converting flac to wav...;
  for i in *.flac; do
    sox "$i" "${i/.flac/}".wav;
    done
  echo Removing flac files...
  for i in *.flac; do
    rm "$i";
    done
  echo Changing back to original directory;
  cd ../..
  done
