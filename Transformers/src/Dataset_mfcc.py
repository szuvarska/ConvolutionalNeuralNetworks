from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset
import torchaudio

import torchaudio.transforms as T


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        max_len: int = 16000,
        transform=None,
        mode: str = "original",
        commands: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.max_len = max_len
        self.transform = transform
        self.mode = mode
        if self.mode not in ["original", "modified"]:
            self.mode = "original"

        # Precompute MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
        )

        all_labels = sorted({p.name for p in self.root_dir.iterdir() if p.is_dir()})

        if self.mode == "modified":
            if commands is None:
                commands = [
                    "yes",
                    "no",
                    "up",
                    "down",
                    "left",
                    "right",
                    "on",
                    "off",
                    "stop",
                    "go",
                    "silence",
                ]
            self.labels = sorted(commands + ["unknown"])
        else:
            self.labels = all_labels

        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        self.samples = []
        for label in all_labels:
            label_dir = self.root_dir / label
            for wav_file in label_dir.glob("**/*.wav"):
                if self.mode == "original":
                    target_label = label
                else:
                    target_label = label if label in commands else "unknown"
                self.samples.append((wav_file, self.label_to_index[target_label]))

    def __len__(self):
        return len(self.samples)

    def pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(1) > self.max_len:
            return waveform[:, : self.max_len]
        elif waveform.size(1) < self.max_len:
            padding = torch.zeros((waveform.size(0), self.max_len - waveform.size(1)))
            return torch.cat([waveform, padding], dim=1)
        return waveform

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        audio_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        waveform = self.pad_or_truncate(waveform)

        features = self.mfcc_transform(waveform)  # (batch, n_mfcc, time)

        if self.transform:
            features = self.transform(features)

        return features, label

    @property
    def class_to_idx(self):
        return self.label_to_index
