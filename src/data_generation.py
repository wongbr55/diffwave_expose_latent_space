from diffwave.inference import predict
from gtts import gTTS
from pydub import AudioSegment
import io
import os
import torch

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
        

    