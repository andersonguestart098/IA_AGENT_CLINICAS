from bark import generate_audio
from scipy.io.wavfile import write as write_wav
import os

# Caminho fixo para salvar os áudios
OUTPUT_PATH = "outputs/voz.wav"

def gerar_voz(texto: str, voz: str = "en_speaker_2") -> str:
    audio_array = generate_audio(texto, history_prompt=voz)
    write_wav(OUTPUT_PATH, rate=22050, data=audio_array)
    return OUTPUT_PATH
