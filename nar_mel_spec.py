import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torchaudio

import numpy as np
from data_loading import NeuralLatentWERDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import pdb
import json
import os
import whisper
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, process_words
from scipy.io.wavfile import write

from model_training import WHISPER_MODEL_PATH, WHISPER_SAMPLE_RATE, DIFFWAVE_MODEL_PATH, SAMPLE_RATE
from diffwave.inference import predict

MEL_SPEC_COL_LEN = 80

def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, d_model):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, x, input_lengths):
        B, T, _ = x.shape
        # boolean mask for padding
        padding_mask = torch.arange(T, device=x.device)[None, :] >= input_lengths.to(x.device)[:, None]
        x = self.pos_enc(self.in_proj(x))
        memory = self.transformer(x, src_key_padding_mask=padding_mask)
        return memory, padding_mask


class NonAutoregressiveMelDecoder(nn.Module):
    def __init__(self, d_model, mel_dim):
        super().__init__()
        self.length_predictor = nn.Linear(d_model, 1)
        self.mel_proj = nn.Linear(d_model, mel_dim)
        self.pos_emb = SinusoidalPositionalEncoding(d_model)
        self.refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )

    def upsample(self, memory, target_lengths):
        B, S, D = memory.shape
        max_len = target_lengths.max()

        memory = memory.permute(0, 2, 1)  # (B, D, S)

        expanded = F.interpolate(
            memory,
            size=max_len,
            mode="linear",
            align_corners=False
        )

        return expanded.permute(0, 2, 1)

    def forward(self, memory, target_lengths=None):
        B, S, D = memory.shape

        # ---- length prediction ----
        pooled = memory.mean(dim=1)
        pred_len = self.length_predictor(pooled).squeeze(-1)

        if target_lengths is not None:
            lengths = target_lengths
        else:
            lengths = pred_len.clamp(min=1).long()

        # ---- upsample properly ----
        T = lengths.max().item()
        pos = self.pos_emb.pe[:, :T, :]  # (1, T, D)
        expanded = self.upsample(memory, lengths)
        expanded = expanded + pos
        mask = torch.arange(T, device=lengths.device)[None, :] >= lengths[:, None]
        expanded = self.refiner(expanded, src_key_padding_mask=mask)

        # ---- predict mel ----
        mel = self.mel_proj(expanded)

        return mel, pred_len


class MelSpecModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: NonAutoregressiveMelDecoder, d_model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, tgt_lengths=None):
        memory, _ = self.encoder(src, src_lengths)

        # Decoder should output full sequence directly
        if tgt_lengths is not None:
            mel_preds, pred_lengths = self.decoder(memory, tgt_lengths)
        else:
            mel_preds, pred_lengths = self.decoder(memory)

        return mel_preds, pred_lengths


def collate_fn_mel_spec(batch):

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
    targets = [
        torch.from_numpy(b["mel_spec"]).float().transpose(0, 1)
        for b in batch
    ]
    
    target_lengths = torch.tensor(
        [y.size(0) for y in targets],
        dtype=torch.long
    )
    # (B, T_out_max, output_dim) or (B, T_out_max)
    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=0.0
    ).contiguous()
    
    
    lengths = torch.tensor([b["mel_spec"].shape[1] for b in batch])
    max_len = lengths.max()
    stop_targets = torch.zeros(len(batch), max_len)
    for i, L in enumerate(lengths):
        stop_targets[i, L-1] = 1.0
    
    return inputs, input_lengths, targets, target_lengths, [b["wav_form"] for b in batch], stop_targets

def train_loop_mel_spec(
    train_loader, model, device, optimizer, sampler, epoch
):
    model.train()
    sampler.set_epoch(epoch)

    train_loss_sum = torch.tensor(0.0, device=device)
    train_reg_sum  = torch.tensor(0.0, device=device)
    train_length_sum = torch.tensor(0.0, device=device)
    train_count    = torch.tensor(0.0, device=device)

    for inputs, input_lengths, targets, target_lengths, wavs, stop_targets in train_loader:

        inputs = inputs.to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        stop_targets = stop_targets.to(device)

        optimizer.zero_grad()

        # ---- Forward (single pass) ----
        preds, pred_lengths = model(inputs, input_lengths, target_lengths)

        B, T, D = targets.shape

        # ---- Mask ----
        mask = torch.arange(T, device=device)[None, :] < target_lengths[:, None]
        num_valid = mask.sum()

        # ---- Mel Loss ----
        reg_loss = F.mse_loss(
            preds[mask],
            targets[mask],
            reduction="sum"
        )

        # ---- Length Loss ----
        length_loss = F.l1_loss(
            pred_lengths.float(),
            target_lengths.float(),
            reduction="sum"
        )

        loss = reg_loss + length_loss

        loss.backward()
        optimizer.step()

        train_reg_sum  += reg_loss.detach()
        train_loss_sum += loss.detach()
        train_length_sum += length_loss.detach()
        train_count    += num_valid

    dist.all_reduce(train_loss_sum)
    dist.all_reduce(train_reg_sum)
    dist.all_reduce(train_length_sum)
    dist.all_reduce(train_count)

    return (
        train_loss_sum.item(),
        train_reg_sum.item(),
        train_length_sum.item(),
        train_count.item()
    )
@torch.no_grad()
def val_loop_mel_spec(val_loader, model, device, sampler, epoch):
    model.eval()
    stt_model = whisper.load_model(WHISPER_MODEL_PATH)
    sampler.set_epoch(epoch)

    val_loss_sum = torch.tensor(0.0, device=device)
    val_reg_sum  = torch.tensor(0.0, device=device)
    val_length_sum = torch.tensor(0.0, device=device)
    val_count    = torch.tensor(0.0, device=device)
    total_errors = torch.zeros(1, device=device)
    loop_iteration = 0
    total_gt_words = torch.zeros(1, device=device)
    for inputs, input_lengths, targets, target_lengths, wavs, stop_targets in val_loader:
        if rank == 0:
            print(f"Val loop iteration {loop_iteration}")
        loop_iteration += 1
        inputs = inputs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        stop_targets = stop_targets.to(device)

        preds, pred_lengths = model(inputs, input_lengths)
        B, T, D = targets.shape
                    
        # preds = prepare_decoder_preds_for_diffwave(preds)
        # PERFORM WER CALC ON PREDICTED
        for i in range(0, preds.shape[0]):
            audio, __, __ = predict(preds[i].transpose(0, 1), DIFFWAVE_MODEL_PATH)
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
        
        T_pred = preds.size(1)
        mask = torch.arange(T_pred, device=device)[None, :] < (target_lengths[:, None])
        num_valid = mask.sum()
        
        reg_loss = F.mse_loss(
            preds[mask],
            targets[:, :T_pred][mask],
            reduction="sum"
        )

        length_loss = F.l1_loss(
            pred_lengths.float(),
            target_lengths.float(),
            reduction="sum"
        )
        loss = reg_loss + length_loss

        val_reg_sum  += reg_loss
        val_length_sum += length_loss
        val_loss_sum += loss
        val_count    += num_valid

    dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_reg_sum,  op=dist.ReduceOp.SUM)
    dist.all_reduce(val_length_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_count,    op=dist.ReduceOp.SUM)
    dist.all_reduce(total_errors, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_gt_words, op=dist.ReduceOp.SUM)

    return (
        val_loss_sum.item(),
        val_reg_sum.item(),
        val_length_sum.item(),
        val_count.item(),
        total_errors,
        total_gt_words
    )
    
def create_mel_model():
        
    INPUT_DIM = 512
    d_model = 128
    encoder = Encoder(INPUT_DIM, 2, 4, d_model)
    decoder = NonAutoregressiveMelDecoder(d_model=d_model, mel_dim=MEL_SPEC_COL_LEN)
    model = MelSpecModel(encoder, decoder, d_model=d_model)
    return model
    

if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_ddp(rank, world_size)
    
    MODEL_NAME = "nar_mel_spec_model"
    INPUT_DIM = 512
    d_model = 128
    num_epochs = 100
    stop_threshold = 10
    batch_size = 128
    device = torch.device(f"cuda:{rank}")
    local_batch_size = batch_size // world_size
    
    encoder = Encoder(INPUT_DIM, 2, 4, d_model)
    decoder = NonAutoregressiveMelDecoder(d_model=d_model, mel_dim=MEL_SPEC_COL_LEN)
    model = MelSpecModel(encoder, decoder, d_model=d_model)
    
    
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank
    )
    optimizer = AdamW(model.parameters(), lr=5e-4)
    
    train_dataset = NeuralLatentWERDataset("/scratch/wongbr55/latent_mel_data", train_or_val=True, latent_timestep=0)
    val_dataset = NeuralLatentWERDataset("/scratch/wongbr55/latent_mel_data", train_or_val=False, latent_timestep=0)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        collate_fn=collate_fn_mel_spec,
        pin_memory=True,
        persistent_workers=True,
        num_workers = 4,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        collate_fn=collate_fn_mel_spec,
        pin_memory=True,
        persistent_workers=True,
        num_workers = 4,
        sampler=val_sampler
    )
    
    train_total_loss = []
    train_reg_loss = []
    train_length_loss = []
    
    val_total_loss = []
    val_reg_loss = []
    val_length_loss = []
    val_wer_score = []
    
    
    os.makedirs(f"/scratch/wongbr55/{MODEL_NAME}", exist_ok=True)
     # use mixed teacher forcing
    initial_ratio = 1.0      # start fully teacher forcing
    final_ratio   = 0.1
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"##############")
            print(f"Starting epoch {epoch}")
        train_loss_sum, train_reg_sum, train_length_sum, train_count = train_loop_mel_spec(train_loader, model, device, optimizer, train_sampler, epoch)
        val_loss_sum, val_reg_sum, val_length_sum, val_count, wer_score_total, total_gt_words = val_loop_mel_spec(val_loader, model, device, val_sampler, epoch)
        
        avg_train_loss = train_loss_sum / train_count
        avg_train_reg = train_reg_sum / train_count
        avg_train_length = train_length_sum / train_count
        
        avg_val_loss = val_loss_sum / val_count
        avg_val_reg = val_reg_sum / val_count
        avg_val_length = val_length_sum / val_count
        
        train_total_loss.append(avg_train_loss)
        train_reg_loss.append(avg_train_reg)
        train_length_loss.append(avg_train_length)
        
        val_total_loss.append(avg_val_loss)
        val_reg_loss.append(avg_val_reg)
        val_length_loss.append(avg_val_length)
        val_wer_score.append(wer_score_total / total_gt_words)
        
        
        if rank == 0:
            torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_total_loss" : train_total_loss,
            "train_reg_loss" : train_reg_loss,
            "train_length_loss" : train_length_loss,
            "val_total_loss" : val_total_loss,
            "val_reg_loss" : val_reg_loss,
            "val_length_loss" : val_length_loss,
            "val_wer_score" : val_wer_score
            }, f"/scratch/wongbr55/{MODEL_NAME}/{MODEL_NAME}_checkpoint_epoch_{epoch}.pt")
