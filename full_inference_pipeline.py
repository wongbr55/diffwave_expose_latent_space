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
from model_mel_spec import create_mel_model
from model_tokenized_latent import create_latent_model, EOS_TOKEN, BOS_TOKEN

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
        [b["wav_form"] for b in batch], stop_targets, [b["sentence"] for b in batch]

@torch.no_grad()
def autoregressive_inference_mel(model, src, src_lengths, tokenized_latent_len, max_len=1000):
    model.eval()
    device = src.device
    B = src.shape[0]

    memory, _ = model.encoder(src, src_lengths)
    ys = torch.zeros(B, 1, MEL_SPEC_COL_LEN, device=device)
    # finished = torch.zeros(B, dtype=torch.bool, device=device)
    gen_lengths = torch.zeros(B, dtype=torch.long, device=device)
    mel_lengths = (tokenized_latent_len * ENCODEC_HOP) // MEL_HOP
    max_steps = min(max_len, mel_lengths.max().item())

    for t in range(max_steps):
        preds, _ = model.decoder(ys, memory)
        next_frame = preds[:, -1:, :]   # (B, 1, mel_dim)

        # which sequences still need frames
        active = (t < mel_lengths)  # (B,)

        # update lengths
        gen_lengths[active] += 1

        # freeze finished sequences (keep last frame)
        next_frame = torch.where(
            active.view(B, 1, 1),
            next_frame,
            ys[:, -1:, :]
        )

        ys = torch.cat([ys, next_frame], dim=1)

    # remove BOS
    ys = ys[:, 1:]

    return ys, gen_lengths

@torch.no_grad()
def autoregressive_inference_latent(model, inputs, input_lengths, max_len=2000, start_token=BOS_TOKEN):
    model.eval()
    B = inputs.size(0)
    device = inputs.device

    memory, _ = model.encoder(inputs, input_lengths)

    # Start with BOS
    decoder_input = torch.full(
        (B, 1, NUM_RESIDUALS),
        start_token,
        device=device,
        dtype=torch.long
    )

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated_lengths = torch.zeros(B, dtype=torch.long, device=device)
    all_preds = []

    for t in range(max_len):
        preds, _ = model.decoder(decoder_input, memory)

        next_logits = preds[:, -1, :, :]
        next_tokens = next_logits.argmax(dim=-1)  # [B, R]

        # who is still active BEFORE this step
        active = ~finished  # [B]
        # detect EOS at this step
        is_eos = (next_tokens == EOS_TOKEN).all(dim=-1)  # [B]

        generated_lengths[active & ~is_eos] += 1

        # mark finished AFTER counting
        finished = finished | is_eos

        # freeze finished sequences
        next_tokens = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_tokens, EOS_TOKEN),
            next_tokens
        )

        decoder_input = torch.cat(
            [decoder_input, next_tokens.unsqueeze(1)],
            dim=1
        )

        if finished.all():
            break
        
    raw_latent_len = generated_lengths * ENCODEC_HOP
    preds = torch.cat(all_preds, dim=1)  # [B, T, R, C]
    
    return preds, generated_lengths, raw_latent_len

# @torch.no_grad()
# def autoregressive_inference_latent(model, inputs, input_lengths, mel_lengths, start_token=0, device="cuda"):
#     """
#     Generate latent tokens autoregressively, using mel lengths to determine sequence length per batch item.
    
#     Args:
#         model: TokenizedAudioModel
#         mel_specs: (B, T_mel, MEL_SPEC_COL_LEN)
#         mel_lengths: (B,) actual lengths from mel generation
#         max_len: optional max length override
#     """
#     model.eval()
#     B = inputs.size(0)
#     device = inputs.device
#     memory, _ = model.encoder(inputs, input_lengths)

#     # Decoder input
#     decoder_input = torch.full((B, 1, NUM_RESIDUALS), start_token, device=device, dtype=torch.float32)
#     all_preds = []
#     all_stop_logits = []

#     # Determine max steps to generate per batch item
#     raw_latent_len = mel_lengths * MEL_HOP
#     tokenized_latent_len = torch.ceil(raw_latent_len / ENCODEC_HOP).long()

#     for t in range(torch.max(tokenized_latent_len)):
#         preds, hidden = model.decoder(decoder_input, memory)
#         stop_logits = model.stop_head(hidden[:, -1, :]).squeeze(-1)

#         next_step_logits = preds[:, -1, :, :]
#         active = (t < tokenized_latent_len).view(B, 1, 1)   # [B,1,1]

#         all_preds.append(next_step_logits.unsqueeze(1) * active.unsqueeze(-1))
#         all_stop_logits.append(stop_logits.unsqueeze(1) * active.squeeze(-1))

#         # Get predicted tokens
#         next_tokens = next_step_logits.argmax(dim=-1).float().unsqueeze(1)  # [B,1,R]

#         next_input = torch.where(
#             active,
#             next_tokens,
#             decoder_input[:, -1:, :]   # keep last token if inactive
#         )

#         decoder_input = torch.cat([decoder_input, next_input], dim=1)

#     preds = torch.cat(all_preds, dim=1)
#     stop_probs = torch.cat(all_stop_logits, dim=1)

#     return preds, stop_probs, tokenized_latent_len, raw_latent_len

@torch.no_grad()
def inference_evaluation(val_loader, latent_model, mel_model, device, encodec_model, latent_timestep, stop_threshold):
    stt_model = whisper.load_model(WHISPER_MODEL_PATH)
    latent_model.eval()
    mel_model.eval()
    loop_iteration = 0
    total_errors = 0
    total_gt_words = 0
    
    for inputs, input_lengths, tokenized_targets, tokenized_target_lengths, mel_targets, mel_target_lengths, wavs, stop_targets, sentence in val_loader:
        print(f"Val loop iteration {loop_iteration}")
        loop_iteration += 1
        inputs = inputs.to(device)
        input_lengths = input_lengths.to(device)
        tokenized_targets = tokenized_targets.to(device)
        tokenized_target_lengths = tokenized_target_lengths.to(device)
        mel_targets = mel_targets.to(device)
        mel_target_lengths = mel_target_lengths.to(device)
        stop_targets = stop_targets.to(device)
        # CONDITIONAL OUTPUT
        # bos = torch.full(
        #     (tokenized_targets.shape[0], 1, NUM_RESIDUALS),
        #     0,
        #     device=tokenized_targets.device
        # )
        # decoder_input = torch.cat([bos, tokenized_targets[:, :-1, :]], dim=1)
        # latent_preds, __ = latent_model(inputs, input_lengths, decoder_input)
        
        # # get mel spectrogram output
        # bos = torch.zeros(inputs.shape[0], 1, MEL_SPEC_COL_LEN, device=mel_targets.device)
        # decoder_input  = torch.cat([bos, mel_targets[:, :-1]], dim=1)
        # mel_preds, __ = mel_model(inputs, input_lengths, decoder_input)
        
        
        # AUTOREGRESSION
        pdb.set_trace()
        latent_preds, tokenized_latent_lengths, raw_latent_lengths = autoregressive_inference_latent(
            latent_model, inputs, input_lengths
        )
        mel_preds, mel_lengths = autoregressive_inference_mel(mel_model, inputs, input_lengths, tokenized_latent_len=tokenized_latent_lengths)

        
        #  get WER rate
        for i in range(0, latent_preds.shape[0]):
            codes = latent_preds[i].argmax(dim=-1).permute(1, 0).unsqueeze(0).to(device)
            scale = torch.tensor(1.0, device=codes.device)
            encoded_frames = [(codes, scale)]
            gen_wav = encodec_model.decode(encoded_frames)
            gen_wav = gen_wav.cpu()
            gen_wav = convert_audio(gen_wav, encodec_model.sample_rate, SAMPLE_RATE, 1)
            gen_wav = gen_wav.to(device).squeeze(1)
            # sometimes we have to do some extra padding
            # expected_wav_len = int(raw_latent_lengths[i].item())
            # if gen_wav.shape[-1] < expected_wav_len:
            #     pad_amount = expected_wav_len - gen_wav.shape[-1]
            #     gen_wav = F.pad(gen_wav, (0, pad_amount))
            # else:
            #     gen_wav = gen_wav[..., :expected_wav_len]
            
            mel = mel_preds[i].transpose(0, 1)  # (mel_dim, T)
            target_mel_len = gen_wav.shape[-1] // MEL_HOP
            current_mel_len = mel.shape[-1]

            if current_mel_len < target_mel_len:
                pad_amount = target_mel_len - current_mel_len
                mel = F.pad(mel, (0, pad_amount))
            else:
                mel = mel[..., :target_mel_len]
            
            # print(f"Mel length: {mel_preds[i].transpose(0, 1)[..., :int(mel_lengths[i].item())].shape}")
            # print(f"Audio length: {gen_wav.shape}")
            audio, __, __ = predict(
                mel,
                DIFFWAVE_MODEL_PATH,
                inject_latent_var=(latent_timestep, gen_wav)
            )
            # use STT to get words spoken
            # can squeeze because first dimension of audio is 1 (batch size)
            audio = audio.squeeze(0)
            wav_tensor = torch.tensor(wavs[i], dtype=torch.float32)  # convert to float tens
            # gt_audio_16k = torchaudio.functional.resample(wav_tensor, SAMPLE_RATE, WHISPER_SAMPLE_RATE)
            gen_audio_16k = torchaudio.functional.resample(audio, SAMPLE_RATE, WHISPER_SAMPLE_RATE)
            # gt_speech = stt_model.transcribe(
            # gt_audio_16k.cpu().numpy().astype("float32"),
            #     language="en",
            #     task="transcribe"
            #     )["text"]

            gen_speech = stt_model.transcribe(
                gen_audio_16k.cpu().numpy().astype("float32"),
                language="en",
                task="transcribe"
            )["text"]
            out = process_words(sentence, gen_speech)

            errors = out.substitutions + out.deletions + out.insertions
            total_errors += errors
            total_gt_words += len(sentence.split())
            del gen_wav, audio, gen_audio_16k #, gt_audio_16k
            torch.cuda.empty_cache()
        

    return total_errors, total_gt_words
    
if __name__ == "__main__":
    device = torch.device(f"cuda")
    latent_timestep = 3
    INPUT_DIM = 512
    MEL_SPEC_COL_LEN = 80
    TOKEN_CLASSES = 1024
    NUM_RESIDUALS = 4
    PAD_TOKEN = 1024
    MEL_HOP = 256
    ENCODEC_HOP = 320
    batch_size = 128
    stop_thresholds = [0.1 * i for i in range(1, 10)]
    # stop_thresholds = [0.2]
    
    latent_model = create_latent_model().to(device)
    mel_model = create_mel_model().to(device)
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(3)
    encodec_model = encodec_model.to(device)
    # load latent model
    latent_model_name = "token_audio_model_med__weighted_stop_latent_4"
    latent_checkpoint = torch.load(f'/scratch/wongbr55/{latent_model_name}/{latent_model_name}_checkpoint_epoch_99.pt')
    latent_state_dict = {
        k.replace("module.", ""): v
        for k, v in latent_checkpoint['model_state_dict'].items()
    }
    latent_model.load_state_dict(latent_state_dict)
    
    # mel mopdel
    mel_model_name = "mel_spec_model_mixed_weighted_stop_loss"
    mel_checkpoint = torch.load(f'/scratch/wongbr55/{mel_model_name}/{mel_model_name}_checkpoint_epoch_99.pt')
    mel_state_dict = {
        k.replace("module.", ""): v
        for k, v in mel_checkpoint['model_state_dict'].items()
    }

    print(f"Using latent model {latent_model_name} and mel model {mel_model_name}")
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
    
    final_wer = []
    for stop_threshold in stop_thresholds:
        total_errors, total_gt_words = inference_evaluation(val_loader, latent_model, mel_model, device, encodec_model, latent_timestep, stop_threshold)
        curr_wer = total_errors / total_gt_words
        print(f"Final WER for stop threshold {stop_threshold}: {curr_wer}")
        final_wer.append(curr_wer)
    
    results = {"latent_model" : latent_model_name, 
               "mdel_model" : mel_model_name, 
               "stop_thresholds" : stop_thresholds, 
               "wer" : final_wer}
    
    with open(f"/scratch/wongbr55/full_inference_results/full_inference_{latent_model_name}_with_{mel_model_name}.json", "w") as f:
        json.dump(results, f)
        