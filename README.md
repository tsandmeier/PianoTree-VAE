# PianoTree-VAE

This is a fork of the repository for the papaer:

Wang et al., PIANOTREE VAE: Structured Representation Learning for Polyphonic Music, ISMIR 2020.

It was adapted to work with the data structure provided as numpy arrays in the data folder.

To run the program, you have to provide a configuration, depending on which part of the data you want to train:<br />
For example `python3 train_ambrose.py ./model_config.json` to train on the whole dataset, or <br /> `python3 train_ambrose.py ./model_config_low.json`to train only on data with a low rhythmical entropy (see repository datenvorverarbeitung)
