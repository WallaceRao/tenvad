from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import librosa
from ten_vad import TenVad
import io
import json
from typing import List, Tuple

app = FastAPI(title="VAD Audio Detection API", description="Voice Activity Detection API using TenVAD")

def detect_voice_segments(audio_data: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float]]:
    """
    使用TenVAD检测音频中的人声段
    
    Args:
        audio_data: 音频数据数组
        sample_rate: 采样率，默认16000Hz
    
    Returns:
        包含人声段开始和结束时间戳的列表 [(start_time, end_time), ...]
    """
    # 确保音频数据是int16格式
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    # 设置参数：每帧256个样本（约16ms），阈值0.5
    hop_size = 256  # 每帧256个样本，对应约16ms（256/16000=0.016s）
    threshold = 0.5
    
    # 创建TenVAD实例
    vad = TenVad(hop_size, threshold)
    
    # 计算总帧数
    num_frames = len(audio_data) // hop_size
    
    # 存储语音状态的列表（1表示语音，0表示非语音）
    voice_states = []
    
    # 处理每一帧
    for i in range(num_frames):
        frame_data = audio_data[i * hop_size: (i + 1) * hop_size]
        probability, flag = vad.process(frame_data)
        voice_states.append(flag)
    
    # 根据README要求实现连续检测逻辑：
    # 连续50ms检测到人声则认为开始讲话
    # 连续50ms检测不到人声则认为开始静音
    
    # 计算50ms对应的帧数（50ms / 16ms per frame ≈ 3.125 frames，向上取整为4帧）
    frames_for_50ms = int(0.050 / (hop_size / sample_rate))  # 50ms对应的帧数
    
    if frames_for_50ms < 1:
        frames_for_50ms = 1
    
    # 实现语音段检测逻辑
    voice_segments = []
    current_state = 0  # 0表示非语音状态，1表示语音状态
    start_frame = 0
    
    i = 0
    while i < len(voice_states):
        if current_state == 0:  # 当前是非语音状态
            # 寻找连续frames_for_50ms帧都是语音的情况
            consecutive_speech = 0
            start_check = i
            
            while start_check < len(voice_states) and consecutive_speech < frames_for_50ms:
                if voice_states[start_check] == 1:
                    consecutive_speech += 1
                else:
                    consecutive_speech = 0
                start_check += 1
            
            if consecutive_speech >= frames_for_50ms:
                # 找到了连续50ms语音，开始语音段
                current_state = 1
                # 从找到连续语音的位置往前推frames_for_50ms帧作为语音段开始
                start_frame = max(0, start_check - frames_for_50ms)
                i = start_check  # 移动到检查结束的位置
            else:
                i += 1
        
        elif current_state == 1:  # 当前是语音状态
            # 寻找连续frames_for_50ms帧都是非语音的情况
            consecutive_silence = 0
            start_check = i
            
            while start_check < len(voice_states) and consecutive_silence < frames_for_50ms:
                if voice_states[start_check] == 0:
                    consecutive_silence += 1
                else:
                    consecutive_silence = 0
                start_check += 1
            
            if consecutive_silence >= frames_for_50ms:
                # 找到了连续50ms静音，结束语音段
                current_state = 0
                # 结束时间是找到连续静音的位置往前推frames_for_50ms帧
                end_frame = start_check - frames_for_50ms
                
                # 添加语音段
                start_time = start_frame * hop_size / sample_rate
                end_time = end_frame * hop_size / sample_rate
                voice_segments.append((start_time, end_time))
                
                i = start_check  # 移动到检查结束的位置
            else:
                # 没有找到足够的连续静音，继续前进一帧
                i += 1
    
    # 如果音频结束时仍处于语音状态，则将语音段延续到音频结尾
    if current_state == 1:
        start_time = start_frame * hop_size / sample_rate
        end_time = len(audio_data) / sample_rate
        voice_segments.append((start_time, end_time))
    
    return voice_segments


@app.post("/vad_detect/")
async def vad_detect(audio_file: UploadFile = File(...)):
    """
    VAD检测接口
    接收音频文件并返回人声段的时间戳
    """
    try:
        # 读取上传的音频文件
        contents = await audio_file.read()
        
        # 使用soundfile或librosa读取音频
        audio_data, sample_rate = librosa.load(io.BytesIO(contents), sr=16000)
        
        # 执行VAD检测
        voice_segments = detect_voice_segments(audio_data, sample_rate)
        
        # 构造响应结果
        result = {
            "filename": audio_file.filename,
            "sample_rate": sample_rate,
            "voice_segments": [
                {"start": round(start, 3), "end": round(end, 3)} 
                for start, end in voice_segments
            ]
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=400
        )


@app.get("/")
async def root():
    return {"message": "VAD Audio Detection API", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)