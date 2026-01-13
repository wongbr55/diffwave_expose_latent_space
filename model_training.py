import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import numpy as np
from data_loading import construct_latent_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import pdb
import json
import os


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

    def forward(self, inputs, input_lengths, targets=None, target_lengths=None):
        """
        inputs: (B, 512, input_dim)
        targets: (B, T)
        """
    
        if targets is None:
            audio_wavs = []
            generated, gen_lengths = self.generate(inputs, input_lengths)
            for i in range(B):
                audio = generated[i, :gen_lengths[i]]
                audio_wavs.append(audio)
            return audio_wavs

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
        generated = self.bos.expand(B, 1, 1).to(device)
        lengths = torch.ones(B, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for __ in range(max_len):
            preds, stop_logits = self.decoder(
                generated,
                lengths,
                h0
            )

            # Last timestep outputs
            next_val = preds[:, -1:, :]           # (B, 1, 1)
            stop_prob = torch.sigmoid(stop_logits[:, -1])  # (B,)

            generated = torch.cat([generated, next_val], dim=1)

            # Update lengths ONLY for unfinished
            lengths += (~finished).long()

            finished |= stop_prob > stop_threshold
            if finished.all():
                break

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


################################### 
# TRAINING LOOPS
###################################

def train_loop(train_loader, model, device, optimizer, stop_loss_threshold, sampler, epoch):
    model.train()
    sampler.set_epoch(epoch)
    train_loss_sum = 0.0
    train_reg_sum = 0.0
    train_stop_sum = 0.0
    train_count = 0
    loop_counter = 0
    for inputs, input_lengths, targets, target_lengths, stop_targets, lengths in train_loader:
        # if loop_counter > 0:
        #     break
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
            # preds: (B, T_max)
            # stop_logits: (B, T_max)

        T_pred = preds.size(1)

        mask = torch.arange(T_pred, device=device)[None, :] < (lengths[:, None] - 1)
        num_valid = mask.sum().item()

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
            
        train_reg_sum += reg_loss.item()
        train_stop_sum += stop_loss.item()
        train_loss_sum += loss.item()
        train_count += num_valid
    return train_loss_sum, train_reg_sum, train_stop_sum, train_count

def val_loop(val_loader, model, device, stop_loss_threshold, sampler, epoch):
    model.eval()
    sampler.set_epoch(epoch)
    val_loss_sum = 0.0
    val_reg_sum = 0.0
    val_stop_sum = 0.0
    val_count = 0.0
    
    loop_iteration = 0
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths, stop_targets, lengths in val_loader:
            # if loop_iteration > 0:
            #     break
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
            num_valid = mask.sum().item()

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

            val_reg_sum += reg_loss.item()
            val_stop_sum += stop_loss.item()
            val_loss_sum += loss.item()
            val_count += num_valid
            
        return val_loss_sum, val_reg_sum, val_stop_sum, val_count

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
    stop_loss_threshold = 0.5
    batch_size = 128
    
    
    
    
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
    
    
    train_dataset, val_dataset = construct_latent_dataset("/scratch/wongbr55/latent_mel_data", 3)
    if rank == 0:
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"Number of training examples: {len(train_dataset)}")
        print(f"Number of val examples: {len(val_dataset)}")
        print(f"Number of train loop iterations: {len(train_dataset) // batch_size}")
        print(f"Number of val loop iterations: {len(val_dataset) // batch_size}")
    
    
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
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers = 8,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers = 8,
        sampler=val_sampler
    )
    
    train_total_loss = []
    train_reg_loss = []
    train_stop_loss = []
    
    val_total_loss = []
    val_reg_loss = []
    val_stop_loss = []
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"##############")
            print(f"Starting epoch {epoch}")
        train_loss_sum, train_reg_sum, train_stop_sum, train_count = train_loop(train_loader, model, device, optimizer, stop_loss_threshold, train_sampler, epoch)
        val_loss_sum, val_reg_sum, val_stop_sum, val_count = val_loop(val_loader, model, device, stop_loss_threshold, val_sampler, epoch)
        
        avg_train_loss = train_loss_sum / train_count
        avg_train_reg = train_reg_sum / train_count
        avg_train_stop = train_stop_sum / train_count
        
        avg_val_loss = val_loss_sum / val_count
        avg_val_reg = val_reg_sum / val_count
        avg_val_stop = val_stop_sum / val_count
        
        train_total_loss.append(avg_train_loss)
        train_reg_loss.append(avg_train_reg)
        train_stop_loss.append(avg_train_stop)
        
        val_total_loss.append(avg_val_loss)
        val_reg_loss.append(avg_val_reg)
        val_stop_loss.append(avg_val_stop)
        
        if rank == 0:
            torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_total_loss" : train_total_loss,
            "train_reg_loss" : train_reg_loss,
            "train_stop_loss" : train_stop_loss,
            "val_total_loss" : val_total_loss,
            "val_reg_loss" : val_reg_loss,
            "val_stop_loss" : val_stop_loss
            }, f"/scratch/wongbr55/{MODEL_NAME}_checkpoint_epoch_{epoch}.pt")
        

    # torch.save({
    # "epoch": epoch,
    # "model_state_dict": model.state_dict(),
    # "optimizer_state_dict": optimizer.state_dict(),
    # }, f"/scratch/wongbr55/{MODEL_NAME}_checkpoint.pt")
    if rank == 0:
        epochs = range(1, len(train_total_loss) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # Regression loss
        axes[0].plot(epochs, train_reg_loss, label="Train Reg")
        axes[0].plot(epochs, val_reg_loss, label="Val Reg")
        axes[0].set_title("Regression Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Stop loss
        axes[1].plot(epochs, train_stop_loss, label="Train Stop")
        axes[1].plot(epochs, val_stop_loss, label="Val Stop")
        axes[1].set_title("Stop Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].plot(epochs, train_total_loss, label="Train Total Loss")
        axes[2].plot(epochs, val_total_loss, label="Val Total Loss")
        axes[2].set_title("Total Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f"/scratch/wongbr55/{MODEL_NAME}_loss_components.png", dpi=300)
        plt.show()

    
