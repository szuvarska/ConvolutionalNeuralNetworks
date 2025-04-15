import torch


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    return list(waveforms), torch.tensor(labels)
