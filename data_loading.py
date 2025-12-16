import os
import torch
import numpy as np
import json

def load_data(data_dir: str, latent_timestep: int, load_train: bool, load_val: bool):
    """Loads data from save directory into X, Y
    
    Args:
        data_dir (str): path to data
        latent_stimestep (int): latent timestep to use (within [0, 6] with 6 being the most noisy and 0 being the denoised waveform)
    """
    X_train, X_val, Y_train, Y_val = [], [], [], []
    for dir_name in os.listdir(data_dir):
        dst_path = os.path.join(data_dir, dir_name)
        for data_type in os.listdir(dst_path):
            curr_path = os.path.join(dst_path, data_type)
            if "train" in data_type and load_train:
                _load_data_specific(curr_path, X_train, Y_train, latent_timestep)
            elif "val" in data_type and load_val:
                _load_data_specific(curr_path, X_val, Y_val, latent_timestep)
            
    return {"X_train" : X_train, "X_val" : X_val, "Y_train":  Y_train, "Y_val" : Y_val}


def _load_data_specific(data_path: str, X: list, Y: list, latent_timestep: int):

    for filename in os.listdir(data_path):
        if not filename.endswith(".json"):
            continue
        json_path = os.path.join(json_path, filename)
        with open(json_path, 'r') as file:
            data = json.load(file)
            X.append(data["neural_features"])
            Y.append(data["latent_variables"][latent_timestep])