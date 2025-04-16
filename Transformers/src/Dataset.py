from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchaudio


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir: str, max_len: int = 16000, transform=None):
        """
        Initializes the dataset with the given directory, max length, and optional transform.

        Arguments:
            root_dir (str): Path to the root directory containing labeled subdirectories of .wav files.
            max_len (int): The fixed length to pad or truncate the audio to. Default is 16000.
            transform (callable, optional): An optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.max_len = max_len
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

    def pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pads or truncates the waveform to a fixed length (max_len)."""
        if waveform.size(1) > self.max_len:
            return waveform[:, : self.max_len]  # Truncate
        elif waveform.size(1) < self.max_len:
            padding = torch.zeros(
                (waveform.size(0), self.max_len - waveform.size(1))
            )  # Pad
            return torch.cat([waveform, padding], dim=1)
        return waveform

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """Returns a single item (waveform, label) from the dataset."""
        audio_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Apply padding/truncation to waveform
        waveform = self.pad_or_truncate(waveform)

        # Apply any additional transformation
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
