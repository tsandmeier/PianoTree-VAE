import json
import random

import numpy as np
from torch.utils.data import Dataset
import os

from torch.utils.data.dataset import T_co

import utils


class AmbroseDataset(Dataset):
    def __init__(self, source_dir, entropy=None, train=True, fraction=None):

        with open(os.path.join(source_dir, "entropy_dict.json")) as dict_file:
            self.entropy_dict = json.load(dict_file)

        self.entropy_dict = {int(k): float(v) for k, v in self.entropy_dict.items()}
        # normalize the entropies to be between 0 and 1
        max_entropy = max(self.entropy_dict.values())
        self.entropy_dict = {k: (v / max_entropy) for k, v in self.entropy_dict.items()}

        with open(os.path.join(source_dir, "filename_dict.json")) as filename_file:
            self.filename_dict = json.load(filename_file)
        self.filename_dict = {int(k): os.path.join(source_dir, "files", v + ".npy") for k, v in
                              self.filename_dict.items()}

        if entropy is None:
            self.ids = list(self.filename_dict.keys())
        elif entropy == 0:
            self.ids = [key for key in self.entropy_dict.keys() if self.entropy_dict[key] < 0.4]
        elif entropy == 1:
            self.ids = [key for key in self.entropy_dict.keys() if
                        0.33 < self.entropy_dict[key] < 0.66]
        elif entropy == 2:
            self.ids = [key for key in self.entropy_dict.keys() if self.entropy_dict[key] > 0.6]

        if fraction is not None:
            if train:
                self.ids = sorted(self.ids)[:round(len(self.ids) * fraction)]
            else:
                self.ids = sorted(self.ids)[round(len(self.ids) * (1 - fraction)):]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index) -> T_co:
        score = np.load(self.filename_dict.get(self.ids[index]), allow_pickle=True)

        mat_pr = np.zeros((32, 128), int)

        for i in range(len(score)):
            for j in range(len(score[i])):
                pitch = int(score[i][j][0])
                dur = int(score[i][j][1])

                mat_pr[i][pitch] = dur

        piano_grid = utils.target_to_3dtarget(mat_pr, max_note_count=16,
                                              max_pitch=128,
                                              min_pitch=0, pitch_pad_ind=130,
                                              pitch_sos_ind=128,
                                              pitch_eos_ind=129)

        return piano_grid.astype(np.int64)


def partition_data(raw_data, train_percent):
    how_many_numbers = int(round((1 - train_percent) * len(raw_data)))
    shuffled = raw_data[:]
    random.shuffle(shuffled)

    return shuffled[how_many_numbers:], shuffled[:how_many_numbers]


if __name__ == '__main__':
    daten_low = AmbroseDataset("/home/tobias/Music_Generation/ambroseMidis/dataset", entropy=0)
    daten_mid = AmbroseDataset("/home/tobias/Music_Generation/ambroseMidis/dataset", entropy=1)
    daten_high = AmbroseDataset("/home/tobias/Music_Generation/ambroseMidis/dataset", entropy=2, train=False)

    print("LOW : ", len(daten_low))
    print("MID: ", len(daten_mid))
    print("HIGH: ", len(daten_high))

    daten_low = AmbroseDataset("/home/tobias/Music_Generation/ambroseMidis/dataset", entropy=0, fraction=0.75)
    daten_mid = AmbroseDataset("/home/tobias/Music_Generation/ambroseMidis/dataset", entropy=1, fraction=0.95)
    daten_high = AmbroseDataset("/home/tobias/Music_Generation/ambroseMidis/dataset", entropy=2, fraction=0.85,
                                train=False)

    print("LOW : ", len(daten_low))
    print("MID: ", len(daten_mid))
    print("HIGH: ", len(daten_high))
