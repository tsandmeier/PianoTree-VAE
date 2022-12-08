import sys

import music21 as music21
import numpy as np
import torch
from music21 import note

import ambroseDataset
import utils
from model import VAE

# script to load an existing model and show the produced output as a midi file

###############################################################################
# Load config
###############################################################################
config_fn = './model_config.json'
train_hyperparams = utils.load_params_dict('train_hyperparams', config_fn)
model_params = utils.load_params_dict('model_params', config_fn)
data_repr_params = utils.load_params_dict('data_repr', config_fn)
project_params = utils.load_params_dict('project', config_fn)
dataset_path = utils.load_dataset_path(config_fn)

BATCH_SIZE = train_hyperparams['batch_size']
LEARNING_RATE = train_hyperparams['learning_rate']
DECAY = train_hyperparams['decay']
PARALLEL = train_hyperparams['parallel']
if sys.platform in ['win32', 'darwin']:
    PARALLEL = False
N_EPOCH = train_hyperparams['n_epoch']
CLIP = train_hyperparams['clip']
UP_AUG = train_hyperparams['up_aug']
DOWN_AUG = train_hyperparams['down_aug']
INIT_WEIGHT = train_hyperparams['init_weight']
WEIGHTS = tuple(train_hyperparams['weights'])
TFR1 = tuple(train_hyperparams['teacher_forcing1'])
TFR2 = tuple(train_hyperparams['teacher_forcing2'])

###############################################################################
# model parameter
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(
    max_simu_note=data_repr_params['max_simu_note'],
    max_pitch=data_repr_params['max_pitch'],
    min_pitch=data_repr_params['min_pitch'],
    pitch_sos=data_repr_params['pitch_sos'],
    pitch_eos=data_repr_params['pitch_eos'],
    pitch_pad=data_repr_params['pitch_pad'],
    dur_pad=data_repr_params['dur_pad'],
    dur_width=data_repr_params['dur_width'],
    num_step=data_repr_params['num_time_step'],
    note_emb_size=model_params['note_emb_size'],
    enc_notes_hid_size=model_params['enc_notes_hid_size'],
    enc_time_hid_size=model_params['enc_time_hid_size'],
    z_size=model_params['z_size'],
    dec_emb_hid_size=model_params['dec_emb_hid_size'],
    dec_time_hid_size=model_params['dec_time_hid_size'],
    dec_notes_hid_size=model_params['dec_notes_hid_size'],
    dec_z_in_size=model_params['dec_z_in_size'],
    dec_dur_hid_size=model_params['dec_dur_hid_size'],
    device=device
)

model_dict = torch.load("/home/tobias/Music_Generation/ambroseMidis/models/pgrid-best-valid-model.pt")

model.load_state_dict(model_dict)


dataset = ambroseDataset.AmbroseDataset(
    "/home/tobias/Music_Generation/ambroseMidis/dataset", entropy=2)

pitch_outs, dur_outs, dist = model.forward(torch.tensor(np.array([dataset[700]])), inference=True, sample=False)


def conv_bin_to_int(bin_list):
    return bin_list[0] * 16 + bin_list[1] * 8 + bin_list[2] * 4 + bin_list[3] * 2 + bin_list[4]


def pitches_and_durs_to_midi(pitches, durs):
    new_score = music21.stream.Score()

    for step_idx, step in enumerate(pitches[0]):
        for note_idx, multi_note in enumerate(step):
            if multi_note < 128:
                new_note = note.Note(multi_note)
                new_note.quarterLength = ((conv_bin_to_int(durs[0, step_idx, note_idx, :]) + 1).__int__()) / 4
                new_note.offset = step_idx / 4
                new_score.insert(step_idx / 4, new_note)
    return new_score


midi_file = pitches_and_durs_to_midi(torch.argmax(pitch_outs, dim=3), torch.argmax(dur_outs, dim=4))

midi_file.show()
