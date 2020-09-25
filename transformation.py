import numpy as np
import librosa
import torch

class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1, sr=16000):
        self.time = time
        self.sr = sr

    def __call__(self, data):
        samples = data["input"]
        length = int(self.time * self.sr)
        if length < len(samples):
            data["input"] = samples[:length]
        elif length > len(samples):
            data["input"] = np.pad(samples, (0, length - len(samples)), "constant")
        return data

class ToMelSpectrogram(object):
    """Creates the mel spectrogram from an audio. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32, sr=16000):
        self.n_mels = n_mels
        self.sr = sr

    def __call__(self, data):
        samples = data["input"]
        s = librosa.feature.melspectrogram(samples, sr=self.sr, n_mels=self.n_mels)
        data["input"] = librosa.power_to_db(s, ref=np.max)
        return data

class ToTensorFromSpect(object):
    """Converts into a tensor."""

    def __init__(self, normalize=None):
        self.normalize = normalize

    def __call__(self, data):
        input_tensor = torch.tensor(data["input"], requires_grad=True)
        target_tensor = torch.tensor(data["target"])
        if self.normalize is not None:
            mean, std = self.normalize
            input_tensor -= mean
            input_tensor /= std
        data["input"] = input_tensor
        data["output"] = target_tensor
        return data