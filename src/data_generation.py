from diffwave.inference import predict
from gtts import gTTS
from pydub import AudioSegment
import io
import os
import torch
import numpy as np
import json
from diffwave.inference import predict as diffwave_predict
from diffwave.preprocess import preprocess_directory as mel_spectrogram_proccess
# package required for loading data
import h5py

# DiffWave only works with a specific sample rate
SAMPLE_RATE = 22050

def generate_audio_waveforms(sentences: list[str], output_dir: str):
    """Generates the audio waveforms for ground truth sentences and saves them in output_dir

    Args:
        sentences (list[str]): list of ground truth sentences
        output_dir (str): _description_
    """
    
    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(sentences, start=1):
        # Generate TTS in memory
        tts = gTTS(text=text, lang="en")
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)  # rewind

        # Load MP3 from memory with pydub
        audio = AudioSegment.from_file(mp3_fp, format="mp3")

        # Resample to 22050 Hz and export
        audio = audio.set_frame_rate(SAMPLE_RATE)
        path = os.path.join(output_dir, f"sentence_{i}.wav")
        audio.export(path, format="wav")
        
def generate_latent_and_mel(file_dir: str, save_dir: str, model_dir: str, fast_sampling=True):
    """Generates full dataset of latent variables and saves in save_dir

    :param file_dir: directory containing h5py folder
    :type file_dir: str
    :param save_dir: directory to save loaded
    :type save_dir: str
    :param model_dir: directory to pretrained diffwave model
    :type model_dir: str
    :param fast_sampling: boolean for controlling fast sampling for diffusion, if True diffusion has 6 denoising steps
    :type fast_sampling: bool
    """
    # collect data from file_dir
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(file_dir):
        if filename.endswith(".hdf5"):
            file_path = os.path.join(file_dir, filename)
            curr_dir = f"{save_dir}/{file_path.removesuffix(".hdf5")}"
            os.makedirs(curr_dir, exist_ok=True)
            # load data from file
            data = load_h5py_file(file_path)
            # generate audio waveforms
            generate_audio_waveforms(data["sentence_label"], curr_dir)
            # generate mel spectrograms
            mel_spectrogram_proccess(curr_dir)
            # generate latent variables for each sentence
            for i in range(0, len(data["sentence_label"])):
                # extract latent variables
                mel = np.load(f"{curr_dir}/sentence_{i}.spec.npy")
                mel = torch.from_numpy(mel).float()
                __, __, latent_vars = diffwave_predict(mel, model_dir, fast_sampling=fast_sampling, expose_latent_vars=True)
                latent_vars_serializable = {k: v.tolist() for k, v in latent_vars.items()}
                # extract neural features for sentence i
                # of length (T, 512)
                neural_features = data["neural_features"][i]
                # save latent variables and latent variables
                curr_sentence_data = {
                    "latent_variables": latent_vars_serializable,
                    "neural_features": neural_features.tolist() if isinstance(neural_features, np.ndarray) else neural_features
                }
                with open(os.path.join(curr_dir, f"sentence_{i}_neural_latent_data.json"), "w") as f:
                    json.dump(curr_sentence_data, f, indent=4)
                
# CODE TAKEN FROM KAGGLE TO LOAD DATA
def load_h5py_file(file_path: str):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data