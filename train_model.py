import os
import pathlib
import time
import datetime

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import glob
import json
import random
import csv
import concurrent.futures
from multiprocessing import freeze_support

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import mixed_precision

seed = 6969
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

replays_dir = pathlib.Path("replays")
leaderboards_dir = pathlib.Path("leaderboards")
  
def get_leaderboard_replays():
  
  leaderboard_ids = np.array(tf.io.gfile.listdir(str(replays_dir)))
  random.shuffle(leaderboard_ids)
  # leaderboard_ids = leaderboard_ids[:int(len(leaderboard_ids)*0.1)]

  # val_leaderboard_ids = ["287692"]
  val_leaderboard_ids = leaderboard_ids[:int(len(leaderboard_ids)*0.2)]

  train_data = []
  val_data = []
  for leaderboard_id in leaderboard_ids:
    replay_files = glob.glob(f'{replays_dir}/{leaderboard_id}/*.json')

    if leaderboard_id in val_leaderboard_ids:
      val_data.append((leaderboard_id, replay_files))
    else:
      train_data.append((leaderboard_id, replay_files))

  return train_data, val_data

def read_replay_file(file):
  with open(file, "r") as f:
    file_content = f.read()
    json_content = json.loads(file_content)
    return json_content

def get_replay_notes(replay):
  notes = []
  
  prev_zero_note_time = 0
  prev_one_note_time = 0
  
  for note_info, score, note_time in sorted(replay, key=lambda item: item[2]):
    type = note_info[-1]

    # TODO: use map data for note positions and timings to not have to exclude misses (misses are registered much later, which messes up the timings)
    if score < 1:
      continue
    
    # NOTE: 0-100 score range is rare and often happens for tracking problems that are not important here
    # would be good to replace this with acc component only and potentially learn all both acc and swing angles
    # but need different format replay files for that
    # score = max(0, score - 100)
    
    delta_to_zero = note_time - prev_zero_note_time
    delta_to_one = note_time - prev_one_note_time
    
    if type == "0":
      prev_zero_note_time = note_time
      note = preprocess_note(score, delta_to_zero, delta_to_one, note_info)
      notes.append(note)
    if type == "1":
      prev_one_note_time = note_time
      note = preprocess_note(score, delta_to_one, delta_to_zero, note_info)
      notes.append(note)
  
  return notes

def preprocess_note(score, delta, delta_other, note_info):
  # NOTE: timing increases difficulty not linearly and caps out at ~2 seconds
  # no idea if such parameters can be learned by neural networks without adding scaling like I did right here
  delta = max(0, 1.5 - delta)
  delta_other = max(0, 1.5 - delta_other)

  col_number = int(note_info[0])
  row_number = int(note_info[1])
  direction_number = int(note_info[2])
  color = int(note_info[3])
  color_arr = [0, 0]
  color_arr[color] = 1

  col = [0] * 4
  row = [0] * 3
  row_col = [0] * 4 * 3
  direction = [0] * 10
  col[col_number] = 1
  row[row_number] = 1
  row_col[col_number * 3 + row_number] = 1
  direction[direction_number] = 1

  response = []
  
  response.extend(row_col)
  response.extend(direction)
  response.extend(color_arr)
  response.extend([
    delta,
    delta_other,
    score
  ])
  
  return response

def create_segments(notes):
  if len(notes) < segment_size:
    return [], [], [], []
  
  pre_segments = []
  segments = []
  post_segments = []
  scores = []
  for i in range(len(notes) - segment_size):
    if i % prediction_size != 0:
      continue
    
    pre_slice = notes[i:i+pre_segment_size]
    slice = notes[i+pre_segment_size:i+pre_segment_size+prediction_size]
    post_slice = notes[i+pre_segment_size+prediction_size:i+segment_size]

    # NOTE: using relative score can be good to find relative difficulty of the notes more fairly
    # because good players will always get higher acc and worse players will do badly even on easy patterns

    pre_segment = [np.array(note[:-1]) for note in pre_slice]
    segment = [np.array(note[:-1]) for note in slice]
    post_segment = [np.array(note[:-1]) for note in post_slice]

    score = sum([note[-1]/15 for note in slice])/len(slice)

    pre_segments.append(pre_segment)
    segments.append(segment)
    post_segments.append(post_segment)
    scores.append(score)
    
  return pre_segments, segments, post_segments, scores

def preprocess_leaderboard_replays(leaderboard_replays, print_progress=False):
  asd = []
  count = 0
  skip = False
  
  replays = []
  
  for leaderboard_replay in leaderboard_replays:
    replays.append(read_replay_file(leaderboard_replay))
    
  replays.sort(key=lambda replay: replay["info"]["totalScore"], reverse=True)
  
  for replay in replays:
    
    note_infos = replay["noteInfos"]
    scores = replay["scores"]
    note_times = replay["noteTime"]
    left_handed = replay["info"]["leftHanded"]
    if left_handed:
      continue
      # no worky
      note_infos_mirrored = []
      for note_info in note_infos:
        if(len(note_info) > 4):
          note_infos_mirrored.append(note_info)
          continue
        col = int(note_info[0])
        row = int(note_info[1])
        dir = int(note_info[2])
        color = int(note_info[3])
        new_note_info = f"{3-col}{row}{dir if dir < 2 else (dir + 1 if dir%2 == 0 else -1)}{1-color}"
        note_infos_mirrored.append(new_note_info)
      note_infos = note_infos_mirrored

    if(count > 10):
      break
    
    if len(asd) == 0:
      asd = []
      for values in zip(note_infos, scores, note_times):
        if len(values[0]) > 4 or values[1] < -3:
          continue
        if values[1] <= 0:
          asd.append([values[0], [], 0, 0])
        else:
          asd.append([values[0], [max(0, values[1] - 100)], values[2], 1])
    else:
      indexes = {}
      num_elements = 0
      for values in zip(note_infos, scores, note_times):
        if len(values[0]) > 4 or values[1] < -3:
          continue
        
        num_elements += 1
        if values[0] in indexes:
          indexes[values[0]].append([values[0], values[1], values[2]])
        else:
          indexes[values[0]] = [[values[0], values[1], values[2]]]


      if num_elements < len(asd):
        continue
      try:
        for values in asd:
          info = indexes[values[0]].pop(0)

          if info[1] > 0:
            values[1].append(max(0, info[1] - 100))
            values[2] += info[2]
            values[3] += 1
      except:
        skip = True
        break
      
    count += 1

  if skip:
    return [], [], [], []
  
  if count < 4 and print_progress == True:
    return [], [], [], []
  
  if count < 2 and print_progress == False:
    return [], [], [], []
  
  asd2 = []
  for values in asd:
    if values[3] == 0:
      continue
    values[1].sort(reverse=True)
    top_acc = values[1]
    # top_acc = values[1][1:-1]
    acc = sum(top_acc)/len(top_acc) if len(top_acc) > 0 else 0
    values[2] = values[2]/values[3]
    asd2.append([values[0], acc, values[2]])
  
  notes = get_replay_notes(asd2)
  return create_segments(notes)


def generate_data(leaderboards_replays, num_threads):
  pre_segments = []
  segments = []
  post_segments = []
  scores = []
  
  executor = concurrent.futures.ProcessPoolExecutor(num_threads)
  segments_tasks = [executor.submit(preprocess_leaderboard_replays, leaderboard_replays) for leaderboard_id, leaderboard_replays in leaderboards_replays]
  
  for segments_task in tqdm(segments_tasks):
    pre_segment, segment, post_segment, score = segments_task.result()
    pre_segments.extend(pre_segment)
    segments.extend(segment)
    post_segments.extend(post_segment)
    scores.extend(score)
  
  executor.shutdown()
  return [np.array(pre_segments), np.array(segments), np.array(post_segments)], np.array(scores)

num_threads = 8

pre_segment_size = 3
post_segment_size = 3
prediction_size = 3
segment_size = pre_segment_size + post_segment_size + prediction_size
batch_size = 256

if __name__ == '__main__':
  freeze_support()

  train_data, val_data = get_leaderboard_replays()
  test_data = val_data

  train_x, train_y = generate_data(train_data, num_threads)
  val_x, val_y = generate_data(val_data, num_threads)
  note_size = 26

  pre_input = keras.Input(shape=(pre_segment_size, note_size), dtype="float32")
  pre_layer = layers.Flatten()(pre_input)
  pre_layer = layers.Dense(32, activation="relu")(pre_layer)
  pre_layer = layers.Dense(32, activation="relu")(pre_layer)
  input = keras.Input(shape=(prediction_size, note_size), dtype="float32")
  layer = layers.Flatten()(input)
  layer = layers.Dense(32, activation="relu")(layer)
  layer = layers.Dense(32, activation="relu")(layer)
  post_input = keras.Input(shape=(post_segment_size, note_size), dtype="float32")
  post_layer = layers.Flatten()(post_input)
  post_layer = layers.Dense(32, activation="relu")(post_layer)
  post_layer = layers.Dense(32, activation="relu")(post_layer)
  
  inputs = [pre_input, input, post_input]
  layers2 = [pre_layer, layer, post_layer]
  l = layers.Concatenate()(layers2)
  l = layers.Flatten()(l)
  l = layers.Dense(64, activation="relu")(l)
  l = layers.Dense(64, activation="relu")(l)
  out = layers.Dense(1, activation="linear")(l)
  model = models.Model(inputs=inputs, outputs = out)
  model.summary()

  model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
      loss=tf.keras.losses.MeanAbsoluteError(reduction="sum_over_batch_size"),
      metrics=['mae', 'mse'],
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

  totdiff = 0
  for score in val_y:
    totdiff += max(score - avg, avg - score)
  avgdiff = totdiff/len(val_y)
  print(f"Average diff: {avgdiff}")
  
  log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
  
  history = model.fit(
      train_x,
      train_y,
      validation_data=(val_x, val_y),
      batch_size=batch_size,
      epochs=10,
      shuffle=True,
      callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=20), tensorboard_callback]
  )

  predictions = []
  for leaderboard_id, leaderboard_replays in test_data:
    try:
      pre, curr, post, score = preprocess_leaderboard_replays(leaderboard_replays)

      _predictions = model.predict([np.array(pre), np.array(curr), np.array(post)])

      real_sum = 0
      for prediction in score:
        real_sum += prediction

      real_avg = real_sum/_predictions.size
      real_percentage_score = (real_avg*15+100)/115

      prediction_sum = 0
      for prediction in _predictions:
        prediction_sum += prediction[0]

      avg = prediction_sum/_predictions.size
      percentage_score = (avg*15+100)/115

      predictions.append([f"https://scoresaber.com/leaderboard/{leaderboard_id}", round(percentage_score, 5), round(real_percentage_score, 5), abs(round(real_percentage_score - percentage_score, 5))])
    except KeyboardInterrupt:
      raise
    except Exception as e:
      print(e)
      continue
  with open('predictions.csv', 'w', newline='', encoding='utf-8') as f:
      writer = csv.writer(f)
      header = ["LeaderboardId", "Prediction", "Expected", "Difference"]
      writer.writerow(header)

      for prediction in predictions:
        writer.writerow(prediction)


  model.save('model_sleep')