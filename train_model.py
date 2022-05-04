import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import glob
import json

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import mixed_precision


mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def get_files():
  data_dir = pathlib.Path("replays")

  # leaderboard_ids = np.array(tf.io.gfile.listdir(str(data_dir)))

  files = glob.glob(f'{data_dir}/*/*.txt')
  files = tf.random.shuffle(files)

  train_file_count, val_file_count = np.array([0.8, 0.1]) * len(files) # remaining files are for testing
  
  train_files, val_files, test_files = np.split(files, [int(train_file_count), int(val_file_count + train_file_count)])

  print('Number of total examples:', len(files))
  print('Training set size: ', len(train_files))
  print('Validation set size: ', len(val_files))
  print('Test set size: ', len(test_files))
  # NOTE: probably not needed since it could be better to use segments as splits
  return train_files, val_files, test_files


def read_replay_file(file):
  with open(file, "r") as f:
    file_content = f.read()
    formatted_content = file_content[1:-1].replace("\\\"", "\"")
    json_content = json.loads(formatted_content)
    return json_content


def get_replay_notes(replay):
  scores = replay["scores"]
  note_times = replay["noteTime"]
  note_infos = replay["noteInfos"]
  
  # NOTE: left hand and right hand. Not sure which one is which :lul:
  zero_notes = []
  one_notes = []
  
  prev_zero_note_time = 0
  prev_one_note_time = 0
  
  for score, note_time, note_info in zip(scores, note_times, note_infos):
    type = note_info[-1]

    # TODO: use map data for note positions and timings to not have to exclude misses (misses are registered much later, which messes up the timings)
    if score == 0:
      continue
    
    # NOTE: 0-100 score range is rare and often happens for tracking problems that are not important here
    # would be good to replace this with acc component only and potentially learn all both acc and swing angles
    # but need different format replay files for that
    score = max(0, score - 100)
    
    note_info = note_info
    if type == "0":
      delta = note_time - prev_zero_note_time
      prev_zero_note_time = note_time
      note = preprocess_note(score, delta, note_info)
      zero_notes.append(note)
    if type == "1":
      delta = note_time - prev_one_note_time
      prev_one_note_time = note_time
      note = preprocess_note(score, delta, note_info)
      one_notes.append(note)
  
  return zero_notes, one_notes


def preprocess_note(score, time_to_prev_note, note_info):
  # NOTE: timing increases difficulty not linearly and caps out at ~2 seconds
  # no idea if such parameters can be learned by neural networks without adding scaling like I did right here
  time_to_prev_note = max(0, 1 - time_to_prev_note/2)
  time_to_prev_note = time_to_prev_note * time_to_prev_note  
  
  response = [time_to_prev_note]
  col_number = int(note_info[0])
  row_number = int(note_info[1])
  direction_number = int(note_info[2])
  # position = [0] * 4 * 3
  # position[col_number * 3 + row_number] = 1
  # direction = [0] * 9
  # direction[direction_number] = 1

  note_type = [0] * 4 * 3 * 9
  note_type[direction_number * 9 + col_number * 3 + row_number] = 1

  # TODO: test with single array where every note type has 1 element
  # response.extend(position)
  # response.extend(direction)
  response.extend(note_type)
  
  response.append(score)
  
  return response


def create_segments(notes):
  if len(notes) < segment_size:
    return []
  
  segments = []
  for i in range(len(notes) - segment_size):
    notes_slice = notes[i:i+segment_size]

    # NOTE: using relative score can be good to find relative difficulty of the notes more fairly
    # because good players will always get higher acc and worse players will do badly even on easy patterns

    # tot_score = 0
    # for note in notes_slice:
    #   tot_score += note[-1]
    # avg = tot_score/len(notes_slice)

    segment = [note[:-1] for note in notes_slice]
    
    current_score = notes_slice[pre_segment_size][-1]
    # NOTE: current_score divided by 15 to limit it to 0-1 range
    segments.append((segment, current_score/15))
  return segments


def preprocess_files(files):
  segments = []
  for i, file in enumerate(files):
      if i % 10 == 0:
        print(f"Processing file: {i}")
      
      replay = read_replay_file(file)
      zero_notes, one_notes = get_replay_notes(replay)
      
      zero_segments = create_segments(zero_notes)
      one_segments = create_segments(one_notes)
      segments.extend(zero_segments)
      segments.extend(one_segments)
  
  return segments

def preprocess_dataset(files):
  segments = preprocess_files(files)
  np.random.shuffle(segments)

  data_len = len(segments)
  print(f"Number of segments: {data_len}")
  train_len = int(data_len*0.8)

  train_segments = segments[:train_len]
  val_segments = segments[train_len:]
  

  # NOTE: this is all needed to change the shape of the data for a multi input network
  # TODO: replace this with something readable
  train_segments_formatted = [[] for s in range(segment_size)]
  train_scores = []
  for notes, score in train_segments:
    train_scores.append(score)
    for i in range(segment_size):
      train_segments_formatted[i].append(np.array(notes[i])[..., None])

  val_segments_formatted = [[] for s in range(segment_size)]
  val_scores = []
  for notes, score in val_segments:
    val_scores.append(score)
    for i in range(segment_size):
      val_segments_formatted[i].append(np.array(notes[i])[..., None])


  return [np.array(n) for n in train_segments_formatted], np.array(train_scores), [np.array(n) for n in val_segments_formatted], np.array(val_scores)

train_files, val_files, test_files = get_files()

pre_segment_size = 5
post_segment_size = 5
segment_size = pre_segment_size + post_segment_size + 1
batch_size = 512

train_x, train_y, val_x, val_y = preprocess_dataset(train_files[:10])

# note_shape = (22,1,)
note_shape = (109,1,)

inputs = []
dense = []

# NOTE: testing with more nodes for the current note. Not sure if it even makes sense
for i in range(pre_segment_size):
  input = keras.Input(shape=note_shape, dtype="float32")
  l = layers.Flatten()(input)
  l = layers.Dense(256, activation="relu", use_bias=False)(l)
  l = layers.Dense(256, activation="relu", use_bias=False)(l)
  dense.append(l)
  inputs.append(input)

input = keras.Input(shape=note_shape, dtype="float32")
l = layers.Flatten()(input)
l = layers.Dense(1024, activation="relu", use_bias=False)(l)
l = layers.Dense(1024, activation="relu", use_bias=False)(l)
dense.append(l)
inputs.append(input)

for i in range(post_segment_size):
  input = keras.Input(shape=note_shape, dtype="float32")
  l = layers.Flatten()(input)
  l = layers.Dense(256, activation="relu", use_bias=False)(l)
  l = layers.Dense(256, activation="relu", use_bias=False)(l)
  dense.append(l)
  inputs.append(input)

l = layers.Concatenate()(dense)
l = layers.Dense(1024, activation="relu", use_bias=False)(l)
l = layers.Dense(1024, activation="relu", use_bias=False)(l)
# l = layers.Dense(1024, activation="relu", use_bias=False)(l)
# l = layers.Dense(1024, activation="relu")(l)
out = layers.Dense(1, activation="relu", use_bias=False)(l)

model = models.Model(inputs=inputs, outputs = out)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=['mae', 'mse', 'mean_squared_logarithmic_error'],
    steps_per_execution=16,
)

tot = 0
for score in train_y:
  tot += score
avg = tot/len(train_y)

totdiff = 0
for score in train_y:
  totdiff += max(score - avg, avg - score)
avgdiff = totdiff/len(train_y)

print(f"Average value: {avg}")
print(f"Average diff: {avgdiff}")

history = model.fit(
    train_x,
    train_y,
    validation_data=(val_x, val_y),
    batch_size=batch_size,
    epochs=100,
    shuffle=True,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=5)
)
