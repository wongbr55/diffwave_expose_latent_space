import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset

class NeuralLatentDataset(Dataset):
    
    """If train_or_val is True, it is Train set, otherwise val set
    """
    def __init__(self, train_or_val: bool, data_dir: str, latent_timestep: int):
    
        super().__init__()
        self.files = []
        self.data_dir = data_dir
        self.latent_timestep = latent_timestep
        
        assert latent_timestep <= 5
        assert latent_timestep >= 0
        
        data_type = "data_train" if train_or_val else "data_val"
        for data_date in os.listdir(data_dir):
            date_path = os.path.join(data_dir, data_date)
            if not os.path.isdir(date_path):
                continue

            inter_path = os.path.join(data_date, data_type)
            full_inter_path = os.path.join(data_dir, inter_path)

            if not os.path.isdir(full_inter_path):
                continue

            for file in os.listdir(full_inter_path):
                if "neural_latent_data" in file:
                    self.files.append(os.path.join(inter_path, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        npz_path = os.path.join(self.data_dir, self.files[idx])
        with np.load(npz_path) as data:
            X = data["neural"].astype(np.float32)
            Y = data["latent"][self.latent_timestep].astype(np.float32)
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        return {"inputs": X, "targets": Y, "target_len": Y.shape[0]}


def construct_latent_dataset(data_dir: str, latent_timestep: int):
    
    """Constructs torch Datasets for training and validation splits

    Args:
        data_dir (str): path to data
        latent_stimestep (int): latent timestep to use (within [0, 5] with 5 being the most noisy and 0 being the denoised waveform)
    """
    
    # data_dict = load_latent_neural_data(data_dir, latent_timestep, True, True)
    train_dataset = NeuralLatentDataset(True, data_dir, latent_timestep)
    val_dataset = NeuralLatentDataset(False, data_dir, latent_timestep)
    
    return train_dataset, val_dataset


def load_latent_neural_data(data_dir: str, latent_timestep: int, load_train: bool, load_val: bool):
    """Loads data from save directory into X, Y
    
    ~3.5 minutues to load all of the training data
    
    Args:
        data_dir (str): path to data
        latent_stimestep (int): latent timestep to use (within [0, 5] with 5 being the most noisy and 0 being the denoised waveform)
    """
    X_train, X_val, Y_train, Y_val = [], [], [], []
    for dir_name in os.listdir(data_dir):
        print(f"Loading {dir_name}")
        dst_path = os.path.join(data_dir, dir_name)
        for data_type in os.listdir(dst_path):
            curr_path = os.path.join(dst_path, data_type)
            if "train" in data_type and load_train:
                _load_data_specific(curr_path, X_train, Y_train, latent_timestep)
            elif "val" in data_type and load_val:
                _load_data_specific(curr_path, X_val, Y_val, latent_timestep)
            
    return {"X_train" : X_train, "X_val" : X_val, "Y_train":  Y_train, "Y_val" : Y_val}


def convert_json_to_npy(data_dir: str):
    """Converts data in JSON format to .npy files

    :param data_dir: directory to saved data, assumed to have been constructed from data_generation.py
    :type data_dir: str
    """
    for dir_name in os.listdir(data_dir):
        print(f"Converting {dir_name}")
        dst_path = os.path.join(data_dir, dir_name)
        for data_type in os.listdir(dst_path):
            curr_path = os.path.join(dst_path, data_type)
            for json_file in os.listdir(curr_path):
                if not json_file.endswith(".json"):
                    continue
                sentence_parts = json_file.split("_")
                sentence = "_".join(sentence_parts[:2])
                json_file_path = os.path.join(curr_path, json_file)
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)
                neural_data, latent_vars = json_data["neural_features"], json_data["latent_variables"]
                latent_stacked = [latent_vars[str(k)][0] for k in range(0, 6)] # there are [0, 5] latent vars inclusive
                np_neural_data, np_latent_vars = np.array(neural_data), np.array(latent_stacked)
                save_path = os.path.join(curr_path, f"{sentence}_neural_latent_data.npz")
                np.savez(save_path, neural=np_neural_data, latent=np_latent_vars)
                os.remove(json_file_path)
        

def _load_data_specific(data_path: str, X: list, Y: list, latent_timestep: int):

    for filename in os.listdir(data_path):
        if not filename.endswith(".npz"):
            continue
        npz_path = os.path.join(data_path, filename)
        data = np.load(npz_path)
        X.append(data["neural"])
        Y.append(data["latent"][latent_timestep])


# main block to convert data from JSON to numpy files
# if __name__ == "__main__":
#     convert_json_to_npy("/scratch/wongbr55/latent_mel_data")