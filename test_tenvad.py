import numpy as np
import soundfile as sf
from ten_vad import TenVad
import librosa
# 初始化 VAD（自动加载 libten_vad.so）
vad = TenVad()

# 读取音频（必须 16kHz 采样率，单声道）
data, sr = librosa.load("cosyvoice3_result_henan.wav", sr=16000)
if data.dtype != np.int16:
    data = (data * 32767).astype(np.int16)
print("audio:", data.shape)
print("sr:", sr)
hop_size = 256  # 16 ms per frame
threshold = 0.5
ten_vad_instance = TenVad(hop_size, threshold)  # Create a TenVad instance
num_frames = data.shape[0] // hop_size
# Streaming inference

if 1:
    for i in range(num_frames):
        audio_data = data[i * hop_size: (i + 1) * hop_size]
        out_probability, out_flag = ten_vad_instance.process(audio_data) #  Out_flag is speech indicator (0 for non-speech signal, 1 for speech signal)
        print("[%d] %0.6f, %d" % (i, out_probability, out_flag))
        #f.write("[%d] %0.6f, %d\n" % (i, out_probability, out_flag))

