import torch._dynamo
torch._dynamo.disable()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from data_loading import TokenizedDataset, SAMPLE_RATE
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb
import json
import os
import whisper
from jiwer import wer, process_words

from model_training import WHISPER_MODEL_PATH, WHISPER_SAMPLE_RATE, DIFFWAVE_MODEL_PATH, SAMPLE_RATE
from diffwave.inference import predict
from old_model_mel_spec import create_mel_model
from old_model_tokenized_latent import create_latent_model

from encodec.utils import convert_audio
from encodec import EncodecModel


def collate_combined_model(batch):
    inputs = [
        torch.from_numpy(b["inputs"]).float()
        for b in batch
    ]
    input_lengths = torch.tensor(
        [x.size(0) for x in inputs],
        dtype=torch.long
    )

    # (B, T_in_max, 512)
    inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=0.0
    ).contiguous()

    # -------- decoder targets --------
    tokenized_targets = [
        b["targets"].long().T
        for b in batch
    ]
    
    tokenized_target_lengths = torch.tensor(
        [y.size(0) for y in tokenized_targets],
        dtype=torch.long
    )
    # (B, T_out_max, Token_dim) 
    tokenized_targets = pad_sequence(
        tokenized_targets,
        batch_first=True,
        padding_value = PAD_TOKEN
    ).contiguous()
    
    mel_targets = [
        torch.from_numpy(b["mel_spec"]).float().transpose(0, 1)
        for b in batch
    ]
    
    mel_target_lengths = torch.tensor(
        [y.size(0) for y in mel_targets],
        dtype=torch.long
    )
    # (B, T_out_max, output_dim) or (B, T_out_max)
    mel_targets = pad_sequence(
        mel_targets,
        batch_first=True,
        padding_value=0.0
    ).contiguous()
    
    lengths = torch.tensor([b["targets"].shape[1] for b in batch])
    max_len = lengths.max()
    stop_targets = torch.zeros(len(batch), max_len)
    for i, L in enumerate(lengths):
        stop_targets[i, L-1] = 1.0
    
    return inputs, input_lengths, tokenized_targets, \
        tokenized_target_lengths, mel_targets, mel_target_lengths, \
        [b["wav_form"] for b in batch], stop_targets

@torch.no_grad()
def val_loop(val_loader, latent_model, mel_model, device, encodec_model, latent_timestep):
    stt_model = whisper.load_model(WHISPER_MODEL_PATH)
    latent_model.eval()
    mel_model.eval()
    total_errors = torch.zeros(1, device=device)
    loop_iteration = 0
    total_gt_words = torch.zeros(1, device=device)
    
    for inputs, input_lengths, tokenized_targets, tokenized_target_lengths, mel_targets, mel_target_lengths, wavs, stop_targets in val_loader:
        print(f"Val loop iteration {loop_iteration}")
        loop_iteration += 1
        inputs = inputs.to(device)
        tokenized_targets = tokenized_targets.to(device)
        tokenized_target_lengths = tokenized_target_lengths.to(device)
        mel_targets = mel_targets.to(device)
        mel_target_lengths = mel_target_lengths.to(device)
        stop_targets = stop_targets.to(device)
        # get tokenized latent output
        bos = torch.full(
            (tokenized_targets.shape[0], 1, NUM_RESIDUALS),
            0,
            device=tokenized_targets.device
        )
        decoder_input = torch.cat([bos, tokenized_targets[:, :-1, :]], dim=1)
        latent_preds, __ = latent_model(inputs, input_lengths, decoder_input)
        
        # get mel spectrogram output
        bos = torch.zeros(inputs.shape[0], 1, MEL_SPEC_COL_LEN, device=mel_targets.device)
        decoder_input  = torch.cat([bos, mel_targets[:, :-1]], dim=1)
        mel_preds, __ = mel_model(inputs, input_lengths, decoder_input)
                    
        #  get WER rate
        for i in range(0, latent_preds.shape[0]):
            codes = latent_preds[i].argmax(dim=-1).permute(1, 0).unsqueeze(0).to(device)
            scale = torch.tensor(1.0, device=codes.device)
            encoded_frames = [(codes, scale)]
            gen_wav = encodec_model.decode(encoded_frames)
            gen_wav = gen_wav.cpu()
            gen_wav = convert_audio(gen_wav, encodec_model.sample_rate, SAMPLE_RATE, 1)
            gen_wav = gen_wav[..., :wavs[i].shape[-1]].squeeze(1)
            gen_wav = gen_wav.to(device)
            audio, __, __ = predict(mel_preds[i].transpose(0, 1)[..., :mel_target_lengths[i]], DIFFWAVE_MODEL_PATH, inject_latent_var=(latent_timestep, gen_wav))
            # use STT to get words spoken
            # can squeeze because first dimension of audio is 1 (batch size)
            audio = audio.squeeze(0)
            wav_tensor = torch.tensor(wavs[i], dtype=torch.float32)  # convert to float tens
            gt_audio_16k = torchaudio.functional.resample(wav_tensor, SAMPLE_RATE, WHISPER_SAMPLE_RATE)
            gen_audio_16k = torchaudio.functional.resample(audio, SAMPLE_RATE, WHISPER_SAMPLE_RATE)
            gt_speech = stt_model.transcribe(
            gt_audio_16k.cpu().numpy().astype("float32"),
                language="en",
                task="transcribe"
                )["text"]

            gen_speech = stt_model.transcribe(
                gen_audio_16k.cpu().numpy().astype("float32"),
                language="en",
                task="transcribe"
            )["text"]
            out = process_words(gt_speech, gen_speech)

            errors = out.substitutions + out.deletions + out.insertions
            total_errors += errors
            total_gt_words += len(gt_speech.split())
        

    return total_errors.item(), total_gt_words.item()
    
if __name__ == "__main__":
    device = torch.device(f"cuda")
    latent_timestep = 3
    INPUT_DIM = 512
    MEL_SPEC_COL_LEN = 80
    TOKEN_CLASSES = 1024
    NUM_RESIDUALS = 4
    PAD_TOKEN = 1024
    batch_size = 128
    
    latent_model = create_latent_model().to(device)
    mel_model = create_mel_model().to(device)
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(3)
    encodec_model = encodec_model.to(device)
    # load latent model
    latent_model_name = "token_audio_model_latent_3"
    latent_checkpoint = torch.load(f'/scratch/wongbr55/{latent_model_name}/{latent_model_name}_checkpoint_epoch_99.pt')
    latent_state_dict = {
        k.replace("module.", ""): v
        for k, v in latent_checkpoint['model_state_dict'].items()
    }
    latent_model.load_state_dict(latent_state_dict)
    
    # mel mopdel
    mel_model_name = "test_mel_spec_model"
    mel_checkpoint = torch.load(f'/scratch/wongbr55/{mel_model_name}/{mel_model_name}_checkpoint_epoch_99.pt')
    mel_state_dict = {
        k.replace("module.", ""): v
        for k, v in mel_checkpoint['model_state_dict'].items()
    }

    mel_model.load_state_dict(mel_state_dict)
    
    # get dataset
    val_dataset = TokenizedDataset("/scratch/wongbr55/latent_mel_data", train_or_val=False, latent_timestep=latent_timestep)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_combined_model,
        pin_memory=True,
        persistent_workers=True,
        num_workers = 4
    )
    
    total_errors, total_gt_words = val_loop(val_loader, latent_model, mel_model, device, encodec_model, latent_timestep)
    print(f"Final WER: {total_errors / total_gt_words}")