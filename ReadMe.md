# BeatSaber replays based AI


## Setup
1. Install a newish version of python and tensorflow with gpu support (and other missing packages)
2. Download replays from https://drive.google.com/file/d/1ADXywV7HSkdh_ltiMyRB_b5KUH7PIHqe/view?usp=sharing and extract them to /replays (only a small number of replays is uploaded here to save space)
3. run `python train_model.py`


### Current concept:
Parse replays into segments of X notes and try to predict the accuracy of hitting the last note of the segment, which could then be used to evaluate the difficult of the map

#### Potential easy improvements:
1. Include Y upcoming notes in the segments, because difficult also depends on the upcoming notes
2. Use map data for note timing
3. Move to a notebook for easier development

#### Potential medium improvements:
1. Smaller file size replays for faster file reading
2. Improve data preprocessing performance (different approach/more efficient numpy usage/port to more efficient language: C++, C#)

#### Potential complex improvements:
1. Include movement data in predictions to get both expected acc and speed, angle change... needed to hit the note for much more accurate evaluation of the difficulty of the map
