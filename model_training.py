import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torchaudio

import numpy as np
from data_loading import construct_latent_dataset, NeuralLatentWERDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import pdb
import json
import os
import whisper
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

from diffwave.inference import predict

DIFFWAVE_MODEL_PATH = "/home/wongbr55/diff_brain_decoding/diffwave_expose_latent_space/pretrained_model/diffwave-ljspeech-22kHz-1000578.pt"
WHISPER_MODEL_PATH = "/home/wongbr55/diff_brain_decoding/diffwave_expose_latent_space/pretrained_model/whisper_stt.pt"
SAMPLE_RATE = 22050
WHISPER_SAMPLE_RATE = 16000


def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


###################################
# TOY MODEL
###################################

###################################
# POSITIONAL ENCODING
###################################
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150000):
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


###################################
# ENCODER
###################################
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=2, batch_first=True),
            num_layers=1
        )

    def forward(self, x, input_lengths):
        B, T, _ = x.shape
        # boolean mask for padding
        padding_mask = torch.arange(T, device=x.device)[None, :] >= input_lengths.to(x.device)[:, None]
        x = self.pos_enc(self.in_proj(x))
        memory = self.transformer(x, src_key_padding_mask=padding_mask)
        return memory, padding_mask

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        # self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.rnn = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True, bidirectional=False)
    
    def forward(self, x, input_lengths):
        B, T, __ = x.shape
        x = self.in_proj(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.rnn(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )
        
        return out, (h_n, c_n)


###################################
# RNN DECODER
###################################
class RNNDecoder(nn.Module):
    def __init__(self, d_model, num_layers=1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(d_model, 1)
        self.stop = nn.Linear(d_model, 1)

    def forward(self, tgt, target_lengths, h0):
        x = self.embed(tgt)  # (B, T, d_model)
        packed = nn.utils.rnn.pack_padded_sequence(x, target_lengths.cpu(), batch_first=True, enforce_sorted=False)

        with torch.backends.cudnn.flags(enabled=False):
            packed_out, _ = self.rnn(packed, h0)

        x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        preds = self.out(x).squeeze(-1)
        stop_logits = self.stop(x).squeeze(-1)
        return preds, stop_logits


###################################
# FULL MODEL
###################################
class NeuralToLatentModel(nn.Module):
    
    """NOTE that at inference time for stop output of decoder, need to run through sigmoid
    """
    
    
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.encoder = RNNEncoder(input_dim, d_model)
        self.decoder = RNNDecoder(d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, 1))  # learned start token

    def forward(self, inputs, input_lengths, targets=None, target_lengths=None, max_len=100000, stop_threshold=0.5):
        """
        inputs: (B, 512, input_dim)
        targets: (B, T)
        """
    
        if targets is None:
            B = len(input_lengths)
            latent_vals = []
            generated, gen_lengths = self.generate(inputs, input_lengths, stop_threshold, max_len)
            for i in range(B):
                latent_var = generated[i, :gen_lengths[i]]
                latent_vals.append(latent_var)
            return latent_vals

        _, (h_n, _) = self.encoder(inputs, input_lengths)
        # h_n: (num_layers, B, d_model)

        h0 = h_n.contiguous()
        B = targets.size(0)
        tgt_in = torch.cat(
            [self.bos.expand(B, 1, 1), targets[:, :-1].unsqueeze(-1)],
            dim=1
        )
        preds, stop_logits = self.decoder(
            tgt_in,
            target_lengths - 1,
            h0
        )
        return preds, stop_logits

    @torch.no_grad()
    def generate(self, inputs, input_lengths, stop_threshold=0.5, max_len=100000):
        """Autoregressive inference

        :param inputs: _description_
        :type inputs: _type_
        :param input_lengths: _description_
        :type input_lengths: _type_
        :param h0: _description_
        :type h0: _type_
        :param stop_threshold: _description_, defaults to 0.5
        :type stop_threshold: float, optional
        """
        
        device = inputs.device

        # ----- Encode -----
        _, (h_n, _) = self.encoder(inputs, input_lengths)
        h0 = h_n.contiguous()  # (num_layers, B, d_model)

        B = inputs.size(0)

        # ----- Initialize with BOS -----
        T_max = max_len  # maximum timesteps you want to generate
        generated = torch.zeros(B, T_max + 1, 1, device=device)
        generated[:, 0, :] = self.bos  # initialize first timestep with BOS
        lengths = torch.ones(B, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            preds, stop_logits = self.decoder(
                generated,
                lengths,
                h0
            )

            # Last timestep outputs
            next_val = preds[:, -1:]  # (B, 1, 1)
            generated[:, t, :] = next_val
            stop_prob = torch.sigmoid(stop_logits[:, -1])  # (B,)

            stop_prob = torch.sigmoid(stop_logits[:, -1])
            lengths += (~finished).long()
            finished |= stop_prob > stop_threshold
            if finished.all():
                break
            if rank == 0:
                print(t)

        # Remove BOS
        generated = generated[:, 1:]
        # lengths currently include BOS → subtract 1
        lengths = lengths - 1

        return generated, lengths


###################################
# TOY MODEL
###################################


def collate_fn(batch):
        # -------- encoder inputs --------
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
        torch.from_numpy(b["targets"]).float()
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
    
    
    lengths = torch.tensor([b["target_len"] for b in batch])
    max_len = lengths.max()
    stop_targets = torch.zeros(len(batch), max_len)
    for i, L in enumerate(lengths):
        stop_targets[i, L-1] = 1.0

    return inputs, input_lengths, targets, target_lengths, stop_targets, lengths


def collate_fn_word_eval(batch):

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

    targets = [
        torch.from_numpy(b["wav_form"]).float()
        for b in batch
    ]
    
    target_lengths = torch.tensor(
        [y.size(0) for y in targets],
        dtype=torch.long
    )

    # (B, T_out_max, output_dim) or (B, T_out_max)
    # targets = pad_sequence(
    #     targets,
    #     batch_first=True,
    #     padding_value=0.0
    # ).contiguous()
    
    return inputs, input_lengths, targets, target_lengths, [b["mel_spec"] for b in batch]

################################### 
# TRAINING LOOPS
###################################

def train_loop(train_loader, model, device, optimizer, stop_loss_threshold, sampler, epoch):
    model.train()
    sampler.set_epoch(epoch)
    train_loss_sum = torch.tensor(0.0, device=device)
    train_reg_sum  = torch.tensor(0.0, device=device)
    train_stop_sum = torch.tensor(0.0, device=device)
    train_count    = torch.tensor(0.0, device=device)

    loop_counter = 0
    for inputs, input_lengths, targets, target_lengths, stop_targets, lengths in train_loader:
        if rank == 0:
            print(f"Training loop iteration {loop_counter}")
        loop_counter += 1

        inputs = inputs.to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        stop_targets = stop_targets.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        preds, stop_logits = model(inputs, input_lengths, targets, target_lengths)

        T_pred = preds.size(1)

        mask = torch.arange(T_pred, device=device)[None, :] < (lengths[:, None] - 1)
        num_valid = mask.sum()

        reg_loss = F.mse_loss(
            preds[mask],
            targets[:, 1:T_pred+1][mask],
            reduction="sum"
        )

        last_mask = torch.arange(T_pred, device=device)[None, :] == (lengths[:, None] - 2)

        stop_loss = F.binary_cross_entropy_with_logits(
            stop_logits[last_mask],
            stop_targets[:, 1:T_pred+1][last_mask],
            reduction="sum"
        )

        loss = reg_loss + stop_loss_threshold * stop_loss

        loss.backward()
        optimizer.step()

        train_reg_sum  += reg_loss.detach()
        train_stop_sum += stop_loss.detach()
        train_loss_sum += loss.detach()
        train_count    += num_valid

    dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_reg_sum,  op=dist.ReduceOp.SUM)
    dist.all_reduce(train_stop_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_count,    op=dist.ReduceOp.SUM)

    return (
        train_loss_sum.item(),
        train_reg_sum.item(),
        train_stop_sum.item(),
        train_count.item()
    )


def val_loop(val_loader, model, device, stop_loss_threshold, sampler, epoch):
    model.eval()
    sampler.set_epoch(epoch)

    val_loss_sum = torch.tensor(0.0, device=device)
    val_reg_sum  = torch.tensor(0.0, device=device)
    val_stop_sum = torch.tensor(0.0, device=device)
    val_count    = torch.tensor(0.0, device=device)

    loop_iteration = 0
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths, stop_targets, lengths in val_loader:
            if rank == 0:
                print(f"Val loop iteration {loop_iteration}")
            loop_iteration += 1

            inputs = inputs.to(device)
            targets = targets.to(device)
            stop_targets = stop_targets.to(device)
            lengths = lengths.to(device)

            preds, stop_logits = model(inputs, input_lengths, targets, target_lengths)

            T_pred = preds.size(1)

            mask = torch.arange(T_pred, device=device)[None, :] < (lengths[:, None] - 1)
            num_valid = mask.sum()

            reg_loss = F.mse_loss(
                preds[mask],
                targets[:, 1:T_pred+1][mask],
                reduction="sum"
            )

            last_mask = torch.arange(T_pred, device=device)[None, :] == (lengths[:, None] - 2)

            stop_loss = F.binary_cross_entropy_with_logits(
                stop_logits[last_mask],
                stop_targets[:, 1:T_pred+1][last_mask],
                reduction="sum"
            )

            loss = reg_loss + stop_loss_threshold * stop_loss

            val_reg_sum  += reg_loss
            val_stop_sum += stop_loss
            val_loss_sum += loss
            val_count    += num_valid

    dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_reg_sum,  op=dist.ReduceOp.SUM)
    dist.all_reduce(val_stop_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_count,    op=dist.ReduceOp.SUM)

    return (
        val_loss_sum.item(),
        val_reg_sum.item(),
        val_stop_sum.item(),
        val_count.item()
    )


def eval_model_wer(val_loader, model, device, stop_thershold, sampler, epoch, latent_timestep):
    
    model.eval()
    sampler.set_epoch(epoch)
    wer_score_total = 0
    loop_iteration = 0
    transform = Compose([
                    ToLowerCase(),
                    RemovePunctuation(),
                    RemoveMultipleSpaces(),
                    Strip()
                ])

    stt_model = whisper.load_model(WHISPER_MODEL_PATH)
    num_samples = 0
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths, mel_spec in val_loader:
            if rank == 0:
                print(f"Eval iteration {loop_iteration}")
                loop_iteration += 1
            inputs = inputs.to(device)
            input_lengths = input_lengths.to(device)
            max_len = max(input_lengths)
            
            latent_vars = model(inputs, input_lengths, stop_threshold=stop_thershold, max_len=max_len)
            if rank == 0:
                print(f"Generated latent_vars")
            # either pad or reomove each latent var to target length
            trimmed = [
                lv[:target_lengths[i]]
                for i, lv in enumerate(latent_vars)
            ]
            padded = pad_sequence(trimmed, batch_first=True, padding_value=0.0)
            num_samples += padded.shape[0]

            for i, var in enumerate(padded):
                audio, __, __ = predict(torch.from_numpy(mel_spec[i]), 
                        DIFFWAVE_MODEL_PATH,
                        inject_latent_var=(latent_timestep, var))
                if rank == 0:
                    print(f"Loaded did {i}")
                # use STT to get words spoken
                gt_audio_16k = torchaudio.functional.resample(targets[i], SAMPLE_RATE, WHISPER_SAMPLE_RATE)
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
                if rank == 0:
                    print(f"Resample {i}")
                
                wer_score_total += wer(gt_speech, gen_speech, truth_transform=transform, hypothesis_transform=transform)
                if rank == 0:
                    print(f"Got WER")


    dist.all_reduce(wer_score_total, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
    
    return wer_score_total, num_samples
################################### 
# TRAINING LOOPS
###################################
# requires at least 1 node with 4 GPUs to run
if __name__ == "__main__":
    # PARAMETERS    
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_ddp(rank, world_size)
    
    MODEL_NAME = "toy_model"
    INPUT_DIM = 512
    d_model = 128
    num_epochs = 10
    stop_threshold = 0.8
    batch_size = 128
    latent_timestep = 3
    
    # does not work for arb. len sequences
    # model = model.to(device)
    # model = torch.compile(model, backend="aot_eager", fullgraph=False)
    
    local_batch_size = batch_size // world_size
    device = torch.device(f"cuda:{rank}")
    model = NeuralToLatentModel(INPUT_DIM, d_model)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank
    )
    optimizer = AdamW(model.parameters(), lr=5e-4)
    
    
    # train_dataset, val_dataset = construct_latent_dataset("/scratch/wongbr55/latent_mel_data", latent_timestep)
    # if rank == 0:
    #     print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    #     print(f"Number of training examples: {len(train_dataset)}")
    #     print(f"Number of val examples: {len(val_dataset)}")
    #     print(f"Number of train loop iterations: {len(train_dataset) // batch_size}")
    #     print(f"Number of val loop iterations: {len(val_dataset) // batch_size}")
    
    #######################
    # WER EVAL
    #######################
    val_wer_dataset = NeuralLatentWERDataset("/scratch/wongbr55/latent_mel_data")
    val_wer_sampler = DistributedSampler(
        val_wer_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_wer_loader = DataLoader(
        val_wer_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        collate_fn=collate_fn_word_eval,
        pin_memory=True,
        persistent_workers=True,
        num_workers = 8,
        sampler=val_wer_sampler
    )
    
    wer_rates = []
    for epoch in range(0, num_epochs):
        checkpoint = torch.load(f'/scratch/wongbr55/toy_model_checkpoint_epoch_{epoch}.pt')

        # new_state_dict = {
        #     k.replace("module.", ""): v
        #     for k, v in checkpoint['model_state_dict'].items()
        # }

        model.load_state_dict(checkpoint['model_state_dict'])
        wer_score, num_sentences = eval_model_wer(val_wer_loader, model, device, stop_threshold, val_wer_sampler, epoch, latent_timestep)

        wer_rates.append(wer_score / num_sentences)
    
    with open("f'/scratch/wongbr55/toy_model_wer_rates.json", "w") as f:
        json.dump(wer_rates, f)

    #######################
    # END WER EVAL
    #######################
    
    
    # train_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=True
    # )
    # val_sampler = DistributedSampler(
    #     val_dataset,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=False
    # )
    
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=local_batch_size,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     num_workers = 8,
    #     sampler=train_sampler
    # )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=local_batch_size,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     num_workers = 8,
    #     sampler=val_sampler
    # )
    
    # train_total_loss = []
    # train_reg_loss = []
    # train_stop_loss = []
    
    # val_total_loss = []
    # val_reg_loss = []
    # val_stop_loss = []
    
    # for epoch in range(num_epochs):
    #     if rank == 0:
    #         print(f"##############")
    #         print(f"Starting epoch {epoch}")
    #     train_loss_sum, train_reg_sum, train_stop_sum, train_count = train_loop(train_loader, model, device, optimizer, stop_loss_threshold, train_sampler, epoch)
    #     val_loss_sum, val_reg_sum, val_stop_sum, val_count = val_loop(val_loader, model, device, stop_loss_threshold, val_sampler, epoch)
        
    #     avg_train_loss = train_loss_sum / train_count
    #     avg_train_reg = train_reg_sum / train_count
    #     avg_train_stop = train_stop_sum / train_count
        
    #     avg_val_loss = val_loss_sum / val_count
    #     avg_val_reg = val_reg_sum / val_count
    #     avg_val_stop = val_stop_sum / val_count
        
    #     train_total_loss.append(avg_train_loss)
    #     train_reg_loss.append(avg_train_reg)
    #     train_stop_loss.append(avg_train_stop)
        
    #     val_total_loss.append(avg_val_loss)
    #     val_reg_loss.append(avg_val_reg)
    #     val_stop_loss.append(avg_val_stop)
        
    #     if rank == 0:
    #         torch.save({
    #         "epoch": epoch,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "train_total_loss" : train_total_loss,
    #         "train_reg_loss" : train_reg_loss,
    #         "train_stop_loss" : train_stop_loss,
    #         "val_total_loss" : val_total_loss,
    #         "val_reg_loss" : val_reg_loss,
    #         "val_stop_loss" : val_stop_loss
    #         }, f"/scratch/wongbr55/{MODEL_NAME}_checkpoint_epoch_{epoch}.pt")
        

    # # torch.save({
    # # "epoch": epoch,
    # # "model_state_dict": model.state_dict(),
    # # "optimizer_state_dict": optimizer.state_dict(),
    # # }, f"/scratch/wongbr55/{MODEL_NAME}_checkpoint.pt")
    # if rank == 0:
    #     epochs = range(1, len(train_total_loss) + 1)
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    #     # Regression loss
    #     axes[0].plot(epochs, train_reg_loss, label="Train Reg")
    #     axes[0].plot(epochs, val_reg_loss, label="Val Reg")
    #     axes[0].set_title("Regression Loss")
    #     axes[0].set_xlabel("Epoch")
    #     axes[0].set_ylabel("Loss")
    #     axes[0].legend()
    #     axes[0].grid(True)

    #     # Stop loss
    #     axes[1].plot(epochs, train_stop_loss, label="Train Stop")
    #     axes[1].plot(epochs, val_stop_loss, label="Val Stop")
    #     axes[1].set_title("Stop Loss")
    #     axes[1].set_xlabel("Epoch")
    #     axes[1].set_ylabel("Loss")
    #     axes[1].legend()
    #     axes[1].grid(True)
        
    #     axes[2].plot(epochs, train_total_loss, label="Train Total Loss")
    #     axes[2].plot(epochs, val_total_loss, label="Val Total Loss")
    #     axes[2].set_title("Total Loss")
    #     axes[2].set_xlabel("Epoch")
    #     axes[2].set_ylabel("Loss")
    #     axes[2].legend()
    #     axes[2].grid(True)

    #     plt.tight_layout()
    #     plt.savefig(f"/scratch/wongbr55/{MODEL_NAME}_loss_components.png", dpi=300)
    #     plt.show()

    
