import torch
from torch.utils.data import Dataset

class ContactForceDataSet(Dataset):
    def __init__(self, hall_signals, heat_maps, transform=None, target_transform=None):
        self.hall_signals = hall_signals
        self.heat_maps = heat_maps
        self.transform = transform
        self.target_transform = target_transform
        self.shape = self.hall_signals.shape

    def __len__(self):
        return len(self.hall_signals)

    def __getitem__(self, idx):
        hall_signal = torch.from_numpy(self.hall_signals[idx].astype('float32'))
        heat_map = torch.from_numpy(self.heat_maps[idx].astype('float32'))
        if self.transform:
            hall_signal = self.transform(hall_signal)
        if self.target_transform:
            heat_map = self.target_transform(heat_map)
        return hall_signal, heat_map
