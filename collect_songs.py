#! /usr/bin/python
import librosa

live_file = '/home/pi/gqrx_20211204_223058_92902000.wav'

station_genre = 'alternative'
station_name = '92_9FM'

import time
import soundfile as sf
from datetime import datetime
import os

file_length = None
def load_and_save_wav(filename, station_name, station_genre, secs=30):
  #Load the filenumpy.float64' object cannot be interpreted as an integer
  audio_recording = filename
  
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

  #Save the file back out.
  sf.write(f'./{station_genre}/{station_name}_{datetime.now().strftime("%H_%M_%S")}.wav', data, sr, sf.default_subtype('WAV'))

os.makedirs(station_genre, exist_ok=True)
  
secs = 30
print(f"Starting script to collect songs from {station_name} with {station_genre} label")
while(True):
  time.sleep(secs+2) #2 seconds of buffer time
  load_and_save_wav(live_file,station_genre=station_genre,station_name=station_name, secs=secs)
  


