"""
Adapted from Speech Commands Data Set v0.02 README
"""
import os
import re
import hashlib
import numpy

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
ROOT = "speech_commands_dataset_v2"
TRAINING_LIST = "datalists/numbers/training_list.txt"
TESTING_LIST = "datalists/numbers/testing_list.txt"
VALIDATION_LIST = "datalists/numbers/validation_list.txt"

GENERATE_NOISE = False
NOISE_PROB = 0.12

def which_set(filename, validation_percentage=10, testing_percentage=10):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name.encode('UTF-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result

# LABELS TO CONSIDER
CLASSES = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  #'left', 'right'
  #noise
  #'cat', 'dog', 'bird'
]

# MAIN
with open(TRAINING_LIST, 'w') as training, \
     open(TESTING_LIST, 'w') as testing, \
     open(VALIDATION_LIST, 'w') as validation:
    for class_dir in os.listdir(ROOT):
        class_path = os.path.join(ROOT, class_dir)
        if not os.path.isdir(class_path) or class_dir.startswith('_'):
            continue
        for wav_file in os.listdir(class_path):
            #print(class_path, wav_file)
            if not wav_file.endswith('.wav'):
                continue
            if class_dir not in CLASSES:
                if GENERATE_NOISE:
                    picked = numpy.random.choice([True, False], p=[NOISE_PROB, 1-NOISE_PROB])
                    if not picked:
                        continue
                else:
                    continue
            line_to_write = f"{class_dir}/{wav_file}\n"            
            assignment = which_set(wav_file)
            if assignment == 'validation':
                validation.write(line_to_write)
            elif assignment == 'testing':
                testing.write(line_to_write)
            else:
                training.write(line_to_write)
