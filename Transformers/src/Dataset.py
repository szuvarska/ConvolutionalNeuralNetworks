from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchaudio


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Get all .wav file paths and corresponding labels
        self.samples = []
        self.labels = sorted({p.name for p in self.root_dir.iterdir() if p.is_dir()})
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        for label in self.labels:
            label_dir = self.root_dir / label
            for wav_file in label_dir.glob("**/*.wav"):
                self.samples.append((wav_file, self.label_to_index[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        audio_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
