#! /usr/bin/python
#Don't leave this in 
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import pandas as pd
import librosa
import numpy as np
from datetime import datetime
# import alsaaudio
from subprocess import call

def stft_mean_var(data):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling stft_mean_var")
  stft=librosa.stft(data)
  stft_db=librosa.amplitude_to_db(abs(stft))  
  return np.mean(stft_db), np.var(stft_db)

def spectral_rolloff_mean_var(data, sr):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling spectral_rolloff_mean_var")
  spectral_rolloff=librosa.feature.spectral_rolloff(data+0.01,sr=sr)[0]
  return np.mean(spectral_rolloff), np.var(spectral_rolloff)

def chroma_stft_mean_var(data, sr):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling chroma_stft_mean_var")
  chroma = librosa.feature.chroma_stft(data,sr=sr)
  return np.mean(chroma), np.var(chroma)

def zero_cross_rate_mean_var(data):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling zero_cross_rate_mean_var")
  zero_cross_rate=librosa.zero_crossings(data)
  return np.mean(zero_cross_rate), np.var(zero_cross_rate)
  # print("the numbert of zero_crossings is :", sum(zero_cross_rate))

def mfcc_mean_var(data, sr):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling mfcc_mean_var")
  mfccs = librosa.feature.mfcc(data, sr=sr)

  results = []
  for mfcc in mfccs:
      results.append(np.mean(mfcc))
      results.append(np.var(mfcc))
  # mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
  return results


def harm_perc_mean_var(data):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling harm_perc_mean_var")
  harm, perc = librosa.effects.hpss(data)
  return np.mean(harm), np.var(harm), np.mean(perc), np.var(perc)

def bpm(data, sr):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling bpm")
  return librosa.beat.beat_track(data, sr = sr)[0]

def spectral_centroid_mean_var(data, sr):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling spectral_centroid_mean_var")
  spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
  return np.mean(spectral_centroids), np.var(spectral_centroids)

def spectral_bandwidth_mean_var(data, sr):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling spectral_bandwidth_mean_var")
  spectral_bandwidths = librosa.feature.spectral_bandwidth(data, sr=sr)[0]
  return np.mean(spectral_bandwidths), np.var(spectral_bandwidths)

    
def rms_mean_var(data, sr):
  # print(f"\t{datetime.now().strftime('%H:%M:%S')} - Calling rms_mean_var")
  rmss = librosa.feature.rms(data)[0]
  return np.mean(rmss), np.var(rmss)

file_length = None



import time
def extract_features(filename, label, last=False, secs=30):
  #Load the filenumpy.float64' object cannot be interpreted as an integer
  audio_recording = filename
  
  

  if last:
    global file_length
    if not file_length:
      cur_time = int(time.time())
      data, sr = librosa.load(audio_recording)
      duration = librosa.get_duration(data,sr) 
      file_length = (cur_time, duration)
    else:
      #New duration is equal to the old duration plus the time since it was calcuated
      duration = file_length[1] + int(time.time()) - file_length[0]

    offset = duration-secs if duration > secs else 0
    print(f"\t{datetime.now().strftime('%H:%M:%S')} - loading last {secs} secs starting at {offset} secs")
    data, sr = librosa.load(audio_recording, duration=secs, offset=offset)
  else:
    data, sr = librosa.load(audio_recording, duration=secs)
  length = np.shape(data)[0]
  #Trim silence
  # data, _ = librosa.effects.trim(data)
  #Call feature functions
  
          # stft_mean_var(data), 
  features = [
          # (0,0),# chroma_stft_mean_var(data, sr), 
          # (0,0), #rms_mean_var(data, sr),
          spectral_centroid_mean_var(data, sr), 
          spectral_bandwidth_mean_var(data, sr),
          spectral_rolloff_mean_var(data, sr), 
          zero_cross_rate_mean_var(data), 
          harm_perc_mean_var(data), 
          [bpm(data, sr)], 
          mfcc_mean_var(data, sr)
          ]
          # "stft_mean", "stft_var", 
  headers = [
          # "chroma_stft_mean", "chroma_stft_var", 
          # "rms_mean", "rms_var", 
          "spectral_centroid_mean", "spectral_centroid_var", 
          "spectral_bandwidth_mean", "spectral_bandwidth_var", 
          "rolloff_mean", "rolloff_var", 
          "zero_crossing_rate_mean", "zero_crossing_rate_var",
          "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var",
          "tempo", 
          'mfcc1_mean','mfcc1_var','mfcc2_mean','mfcc2_var','mfcc3_mean',
          'mfcc3_var','mfcc4_mean','mfcc4_var','mfcc5_mean','mfcc5_var',
          'mfcc6_mean','mfcc6_var','mfcc7_mean','mfcc7_var','mfcc8_mean',
          'mfcc8_var','mfcc9_mean','mfcc9_var','mfcc10_mean','mfcc10_var',
          'mfcc11_mean','mfcc11_var','mfcc12_mean','mfcc12_var','mfcc13_mean',
          'mfcc13_var','mfcc14_mean','mfcc14_var','mfcc15_mean','mfcc15_var',
          'mfcc16_mean','mfcc16_var','mfcc17_mean','mfcc17_var','mfcc18_mean',
          'mfcc18_var','mfcc19_mean','mfcc19_var','mfcc20_mean','mfcc20_var' 
          ]
  #Flatten the return values from all the functions to match the headers list
  features = [ n for x in features for n in x ]
  
  headers.insert(0,"length")
  headers.insert(0,"filename")
  headers.append("label")
  features.insert(0, length)
  features.insert(0, filename)
  features.append(label)

  # print(f"{len(headers)}: {len(features)}")

  # print(headers)
  # print(features)

  df = pd.DataFrame([features], columns=headers)

  return df



def use_tflite_model(tflite_file, X_input):
  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(X_input),), dtype=int)
  for i in range(len(X_input)):
    x_input = X_input[i]

    # Check if the input type is quantized, then rescale input data to uint8
    # if input_details['dtype'] == np.uint8:
    #   input_scale, input_zero_point = input_details["quantization"]
    #   test_image = test_image / input_scale + input_zero_point

    # test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    # print("DEBUG",  input_details['shape'])
    interpreter.set_tensor(input_details["index"], [np.float32(x_input)])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    # print(output)
    predictions[i] = output.argmax()

  return predictions, output

def predict_genre(filename, label, tflite_model_file, last=False, secs=10):
  # print(f"{datetime.now().strftime('%H:%M:%S')} - Generating Features")
  df = extract_features(filename, label, last, secs=secs)
  # df = pd.read_csv("archive/Data/features_30_sec.csv")
  # df = df.loc[df['filename'] == filename]
  #Normalize and drop extra fields
  df=df.drop(labels="filename",axis=1)
  X_predict = np.array((df.iloc[:,:-1] - mean) / std)
  # print(df.columns)
  # print(mean.index)

  #Generate a prediction
  # print(X_predict)
  # print(f"{datetime.now().strftime('%H:%M:%S')} - Calling tensorflow model")
  prediction, output = use_tflite_model(tflite_model_file, X_predict)
  #Using the converter/home/pi/gqrx_20211128_173945_94300000.wav generate the text label for the class.
  # class_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
  class_labels = ['ads', 'alternative', 'classic_rock', 'country', 'top_hits']

  class_label = class_labels[prediction[0]]
  print([f"{class_labels[i]} : {int(output[i] *100)}%" for i in range(len(class_labels))])

  return class_label

#Constants
tflite_model_file = 'models/radio_and_ads/radio_and_ads_10s_model.tflite'
label = 'rock'
mean = pd.read_csv("models/radio_and_ads/radio_and_ads_10s_mean.csv", index_col=0, header=None, squeeze=True)
std = pd.read_csv("models/radio_and_ads/radio_and_ads_10s_std.csv", index_col=0, header=None, squeeze=True)

import glob
import os
def test_audio():
  for filename in glob.glob("*.wav"):
    prediction = predict_genre(filename, label, tflite_model_file)
    print(f"{datetime.now().strftime('%H:%M:%S')} - {filename} was predicted to be {prediction}")

def live_predict():
  # live_file = '/home/pi/gqrx_20211208_021310_94295000.wav'
  path = "/home/pi/gqrx_*.wav"
  wave_files = glob.glob(path)
  wave_files.sort(key=os.path.getctime)
  live_file = wave_files[-1]
  print(live_file)
  # m = alsaaudio.Mixer() 
  ad_flag = False
  while(True):
    prediction = predict_genre(live_file, label, tflite_model_file, last=True, secs=10)
    print(f"{datetime.now().strftime('%H:%M:%S')} - {live_file} was predicted to be {prediction}")

    if (prediction == 'ads' and not ad_flag):
      # m.setvolume(20)
      call(["amixer", "-D", "pulse", "sset", "Master", "20%"])
      print("It's an Ad. Turn down the volume!")
      ad_flag = True
    elif prediction != 'ads' and ad_flag:
      # m.setvolume(50)
      call(["amixer", "-D", "pulse", "sset", "Master", "90%"])
      print("Ads are over. Turn up the volume!")
      ad_flag = False




live_predict()

