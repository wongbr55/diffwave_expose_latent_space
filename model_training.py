import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loading import construct_latent_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import pdb



###################################
# TOY MODEL
###################################

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                batch_first=True
            ),
            num_layers=4
        )

    def forward(self, x, input_lengths):
        B, T, _ = x.shape
        padding_mask = torch.arange(T, device=x.device)[None, :] >= input_lengths[:, None]

        x = self.in_proj(x)
        x = self.pos_enc(x)
        memory = self.transformer(x, src_key_padding_mask=padding_mask)

        return memory, padding_mask


# OLD decoder, padding/attention matricies too large
# class Decoder(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.embed = nn.Linear(1, d_model)  # scalar target → d_model

#         self.transformer = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=d_model,
#                 nhead=8,
#                 batch_first=True
#             ),
#             num_layers=4
#         )
#         self.pos_enc = SinusoidalPositionalEncoding(d_model)


#         self.out = nn.Linear(d_model, 1)     # regression output
#         self.stop = nn.Linear(d_model, 1)    # stop logit

#     def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         # tgt: (B, T, 1)
#         tgt = self.embed(tgt)
#         tgt = self.pos_enc(tgt)


#         hidden = self.transformer(
#             tgt=tgt,
#             memory=memory,
#             tgt_mask=tgt_mask,
#             tgt_key_padding_mask=tgt_key_padding_mask,
#             memory_key_padding_mask=memory_key_padding_mask
#         )                                 # (B, T, d_model)

#         preds = self.out(hidden).squeeze(-1)       # (B, T)
#         stop_logits = self.stop(hidden).squeeze(-1)  # (B, T)

#         return preds, stop_logits

class RNNDecoder(nn.Module):
    
    def __init__(self, d_model) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.rnn = nn.GRU(
            d_model,
            d_model,
            num_layers=4,
            batch_first=True
        )
        self.out = nn.Linear(d_model, 1)
        self.stop = nn.Linear(d_model, 1)
    
    def forward(self, tgt, target_lengths, h0):
        # Ensure contiguous
        tgt = tgt.contiguous()
        h0 = h0.contiguous()
        
        # Embed and make contiguous
        x = self.embed(tgt).contiguous()

        # Pack
        packed = nn.utils.rnn.pack_padded_sequence(
            x, target_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Run RNN
        packed_out, _ = self.rnn(packed, h0)

        # Unpack
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        preds = self.out(x).squeeze(-1)
        stop_logits = self.stop(x).squeeze(-1)
        return preds, stop_logits
        

class NeuralToLatentModel(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model)
        self.decoder = RNNDecoder(d_model)
        
        # learnt begining of seq. token
        # since we are dealing with continous "tokens", we learn a valuable starting representation
        self.bos = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, inputs, input_lengths, targets, target_lengths):
        """
        inputs:  (B, 512, input_dim)
        targets: (B, T)   # teacher forcing
        """
        
        # take transformer encoding and perform reduction
        memory, memory_padding_mask = self.encoder(inputs, input_lengths)
        B, T_enc, D = memory.shape

        # Take final timestep of each sequence for h0
        idx = (input_lengths - 1).view(B, 1, 1).expand(B, 1, D)
        enc_final = memory.gather(1, idx).squeeze(1).contiguous()
        h0 = enc_final.unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1).contiguous()

        if targets is not None:
            B, T = targets.shape
            bos = self.bos.expand(B, 1, 1)

            tgt_in = torch.cat([bos, targets[:, :-1].unsqueeze(-1)], dim=1).contiguous()

            preds, stop_logits = self.decoder(
                tgt=tgt_in,
                target_lengths=target_lengths - 1,
                h0=h0,
            )

            return preds, stop_logits
        
    #     return self.generate(inputs, input_lengths, memory, memory_padding_mask)
    
    # # TODO change to work with RNN decoder
    # @torch.no_grad()
    # def generate(self, inputs, input_lengths, memory=None, memory_padding_mask=None, max_len=100000, stop_threshold=0.5):
    #     """
    #     Autoregressive inference, at inference we do not have access to targets so we perform autoregressive latent genereation
    #     """
    #     device = inputs.device
    #     B = inputs.size(0)

    #     if memory is None:
    #         memory, memory_padding_mask = self.encoder(inputs, input_lengths)

    #     # initialize with BOS
    #     generated = self.bos.expand(B, 1, 1)  # (B, 1, 1)
    #     stop_mask = torch.zeros(B, dtype=torch.bool, device=device)

    #     for __ in range(max_len):
    #         T = generated.size(1)
    #         tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
    #         preds, stop_logits = self.decoder(
    #             tgt=generated,
    #             memory=memory,
    #             tgt_mask=tgt_mask,
    #             tgt_padding_mask=None,
    #             memory_padding_mask=memory_padding_mask
    #         )

    #         # take last timestep prediction
    #         next_step = preds[:, -1].unsqueeze(-1)  # (B, 1)
    #         generated = torch.cat([generated, next_step.unsqueeze(-1)], dim=1)  # (B, T+1, 1)

    #         # check stop logits
    #         stop_prob = torch.sigmoid(stop_logits[:, -1])
    #         stop_mask |= stop_prob > stop_threshold

    #         if stop_mask.all():
    #             break

    #     return generated[:, 1:, 0]  # remove BOS, shape: (B, T_generated)

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

def train_loop(train_loader, model, device, optimizer, stop_loss_threshold):
    model.train()
    train_loss_sum = 0.0
    train_reg_sum = 0.0
    train_stop_sum = 0.0
    train_count = 0
    for inputs, input_lengths, targets, target_lengths, stop_targets, lengths in train_loader:
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
            stop_targets[last_mask],
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

def val_loop(val_loader, model, device, stop_loss_threshold):
    model.eval()
    val_loss_sum = 0.0
    val_reg_sum = 0.0
    val_stop_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths, stop_targets, lengths in val_loader:
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
                stop_targets[last_mask],
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
    
    torch.backends.cudnn.enabled = False
    INPUT_DIM = 512
    num_epochs = 10
    model = NeuralToLatentModel(INPUT_DIM, 256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    stop_loss_threshold = 0.1
    batch_size = 8
    
    train_dataset, val_dataset = construct_latent_dataset("/scratch/wongbr55/latent_mel_data", 3)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    train_total_loss = []
    train_reg_loss = []
    train_stop_loss = []
    
    val_total_loss = []
    val_reg_loss = []
    val_stop_loss = []
    
    for epoch in range(num_epochs):
        train_loss_sum, train_reg_sum, train_stop_sum, train_count = train_loop(train_loader, model, device, optimizer, stop_loss_threshold)
        val_loss_sum, val_reg_sum, val_stop_sum, val_count = val_loop(val_loader, model, device, stop_loss_threshold)
        
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
    

    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, "/scratch/wongbr55/toy_model_checkpoint.pt")

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
    plt.savefig("/scratch/wongbr55/toy_model_loss_components.png", dpi=300)
    plt.show()

    
