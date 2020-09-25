import librosa
import numpy as np
import shelve
import os

ROOT = 'speech_commands_dataset_v2'
WAV_DURATION_SEC = 1
SR = 16000
N_MELS = 32

db = shelve.open(ROOT + '.cache')
for class_dir in os.listdir(ROOT):
    class_path = os.path.join(ROOT, class_dir)
    if not os.path.isdir(class_path):
        continue
    for wav_file in os.listdir(class_path):
        #print(class_path, wav_file)
        if not wav_file.endswith('.wav'):
            continue
        
        dbkey = f"{class_dir}/{wav_file}"
        path = os.path.join(ROOT, class_dir, wav_file)
        
        # Load audio
        samples, sample_rate = librosa.load(path, SR)
        
        # Fix audio length to 1 sec
        length = int(WAV_DURATION_SEC * SR)
        if length < len(samples):
            samples = samples[:length]
        elif length > len(samples):
            samples = np.pad(samples, (0, length - len(samples)), "constant")
        
        # Calculate mel spectrogram
        ms = librosa.feature.melspectrogram(samples, sr=SR, n_mels=N_MELS)
        msdb = librosa.power_to_db(ms, ref=np.max)
        
        # Save to Shelf (not yet tensor!)
        db[dbkey] = msdb
    print('Processed %s' % class_path)

db.close()
