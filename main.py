from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import soundfile as sf
import librosa
from ten_vad import TenVad
import io
import json
from typing import List, Tuple
import pytsmod

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


def get_silence_segments(audio_duration: float, voice_segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    获取静音段的时间戳列表
    
    Args:
        audio_duration: 音频总时长
        voice_segments: 人声段列表
    
    Returns:
        静音段列表 [(start_time, end_time), ...]
    """
    silence_segments = []
    
    # 如果没有人声段，整个音频都是静音
    if not voice_segments:
        return [(0.0, audio_duration)]
    
    # 处理开头的静音段
    if voice_segments[0][0] > 0:
        silence_segments.append((0.0, voice_segments[0][0]))
    
    # 处理人声段之间的静音段
    for i in range(len(voice_segments) - 1):
        silence_start = voice_segments[i][1]
        silence_end = voice_segments[i + 1][0]
        if silence_end > silence_start:  # 确保存在静音段
            silence_segments.append((silence_start, silence_end))
    
    # 处理结尾的静音段
    if voice_segments[-1][1] < audio_duration:
        silence_segments.append((voice_segments[-1][1], audio_duration))
    
    return silence_segments


def adjust_audio_length(audio_data: np.ndarray, sample_rate: int, target_duration: float, voice_segments: List[Tuple[float, float]]) -> np.ndarray:
    """
    调整音频长度到指定目标时长
    
    Args:
        audio_data: 原始音频数据
        sample_rate: 采样率
        target_duration: 目标时长（秒）
        voice_segments: 人声段列表
    
    Returns:
        调整后的音频数据
    """
    original_duration = len(audio_data) / sample_rate
    
    # 计算需要裁剪或扩充的时长
    duration_diff = original_duration - target_duration
    
    if abs(duration_diff) < 0.1:  # 如果差异很小，直接返回原音频
        return audio_data
    
    if duration_diff > 0:  # 需要裁剪
        return _adjust_audio_shorter(audio_data, sample_rate, target_duration, voice_segments, duration_diff)
    else:  # 需要扩充
        return _adjust_audio_longer(audio_data, sample_rate, target_duration, voice_segments, abs(duration_diff))


def _adjust_audio_shorter(audio_data: np.ndarray, sample_rate: int, target_duration: float, 
                         voice_segments: List[Tuple[float, float]], duration_to_remove: float) -> np.ndarray:
    """
    将音频缩短到指定时长
    """
    total_duration = len(audio_data) / sample_rate
    # 这里允许增加200ms 缓解短句子场景加速太多音质下降
    if (total_duration - target_duration) > 0.2:
       target_duration = target_duration + 0.2
       print("adjust total duration to:", target_duration)
    # 获取静音段
    silence_segments = get_silence_segments(total_duration, voice_segments)
    
    # 计算总静音时长
    total_silence_duration = sum(seg[1] - seg[0] for seg in silence_segments)
    if total_silence_duration <= 0:  # 如果没有静音段，使用PV-TSM算法
        # 使用pytsmod的PV-TSM算法将音频调整为目标长度
        ratio = target_duration / total_duration
        adjusted_audio = pytsmod.wsola(audio_data.astype(np.float32), 
                                              ratio)
        return adjusted_audio.astype(audio_data.dtype)
    
    # 检查是否可以通过调整静音段来达到目标长度
    if total_silence_duration >= duration_to_remove:
        # 情况a: 按比例计算每段静音裁剪后的长度
        new_silence_segments = []
        for silence_seg in silence_segments:
            seg_duration = silence_seg[1] - silence_seg[0]
            # 每段静音裁剪后的长度 = 原长度 - (原长度 * 裁剪总量/总静音长度)
            new_duration = seg_duration - (seg_duration * duration_to_remove / total_silence_duration)
            new_duration = max(0, new_duration)  # 确保不为负数
            new_silence_segments.append(new_duration)
    else:
        # 情况b: 即使删除全部静音也超过目标长度，裁剪90%的静音
        new_silence_segments = []
        for silence_seg in silence_segments:
            seg_duration = silence_seg[1] - silence_seg[0]
            new_duration = seg_duration * 0.1  # 保留10%的静音
            new_silence_segments.append(new_duration)
    
    # 重建音频
    reconstructed_audio = _reconstruct_audio_with_new_silence(audio_data, sample_rate, voice_segments, 
                                             silence_segments, new_silence_segments)

    # 如果重建后仍未达到目标长度，使用PV-TSM算法
    current_duration = len(reconstructed_audio) / sample_rate
    if current_duration > target_duration:
        ratio = target_duration / current_duration
        adjusted_audio = pytsmod.wsola(reconstructed_audio.astype(np.float32), ratio)
        return adjusted_audio.astype(audio_data.dtype)


def _adjust_audio_longer(audio_data: np.ndarray, sample_rate: int, target_duration: float, 
                        voice_segments: List[Tuple[float, float]], duration_to_add: float) -> np.ndarray:
    """
    将音频延长到指定时长
    """
    total_duration = len(audio_data) / sample_rate
    # 如果target_duration  < 1.4s，直接在当前音频前后添加静音 延长到target_duration秒
    if target_duration < 1.4:
        padding_duration = target_duration - total_duration
        padding_samples = int(padding_duration * sample_rate)
        padding = np.zeros(padding_samples, dtype=audio_data.dtype)
        print("add padding directly")
        return np.concatenate((padding, audio_data, padding))
    
    # 获取静音段
    
    silence_segments = get_silence_segments(total_duration, voice_segments)
    
    # 计算总静音时长
    total_silence_duration = sum(seg[1] - seg[0] for seg in silence_segments)
    
    if total_silence_duration <= 0:  # 如果没有静音段，使用PV-TSM算法
        # 使用pytsmod的PV-TSM算法将音频调整为目标长度
        ratio = target_duration / total_duration
        adjusted_audio = pytsmod.wsola(audio_data.astype(np.float32), 
                                              ratio)
        return adjusted_audio.astype(audio_data.dtype)
    
    # 检查是否可以通过调整静音段来达到目标长度
    if total_silence_duration >= duration_to_add:
        # 情况a: 按比例计算每段静音扩充后的长度
        new_silence_segments = []
        for silence_seg in silence_segments:
            seg_duration = silence_seg[1] - silence_seg[0]
            # 每段静音扩充后的长度 = 原长度 + (原长度 * 扩充量/总静音长度)
            new_duration = seg_duration + (seg_duration * duration_to_add / total_silence_duration)
            new_silence_segments.append(new_duration)
    else:
        # 情况b: 静音不够用，将每段静音时长调整为当前的2倍
        new_silence_segments = []
        for silence_seg in silence_segments:
            seg_duration = silence_seg[1] - silence_seg[0]
            new_duration = seg_duration * 2.0
            new_silence_segments.append(new_duration)
    
    # 重建音频
    reconstructed_audio = _reconstruct_audio_with_new_silence(audio_data, sample_rate, voice_segments, 
                                                           silence_segments, new_silence_segments)
    
    # 如果重建后仍未达到目标长度，使用PV-TSM算法
    current_duration = len(reconstructed_audio) / sample_rate
    if current_duration < target_duration:
        ratio = target_duration / current_duration
        adjusted_audio = pytsmod.wsola(reconstructed_audio.astype(np.float32), ratio)
        return adjusted_audio.astype(audio_data.dtype)
    
    return reconstructed_audio


def _reconstruct_audio_with_new_silence(audio_data: np.ndarray, sample_rate: int, 
                                      voice_segments: List[Tuple[float, float]], 
                                      original_silence_segments: List[Tuple[float, float]],
                                      new_silence_durations: List[float]) -> np.ndarray:
    """
    根据新的静音段长度重建音频
    """
    # 提取所有人声片段
    voice_clips = []
    for start_time, end_time in voice_segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        # 确保索引在范围内
        start_sample = max(0, min(start_sample, len(audio_data)))
        end_sample = max(0, min(end_sample, len(audio_data)))
        if end_sample > start_sample:
            voice_clips.append(audio_data[start_sample:end_sample])
    
    # 创建静音片段
    silence_clips = []
    for duration in new_silence_durations:
        num_samples = int(duration * sample_rate)
        if num_samples > 0:
            silence_clip = np.zeros(num_samples, dtype=audio_data.dtype)
            silence_clips.append(silence_clip)
    
    # 拼接音频：静音-语音-静音-语音...
    reconstructed_audio_list = []
    
    # 确定拼接顺序
    if original_silence_segments and voice_segments:
        # 检查哪个先出现：静音还是语音
        if original_silence_segments[0][0] < voice_segments[0][0]:
            # 静音在前
            for i in range(max(len(silence_clips), len(voice_clips))):
                if i < len(silence_clips):
                    reconstructed_audio_list.append(_apply_fade(silence_clips[i], fade_type='out'))
                if i < len(voice_clips):
                    reconstructed_audio_list.append(_apply_fade(voice_clips[i], fade_type='both'))
                # 如果语音比静音多（例如音频开头没有静音）
                if i >= len(silence_clips) and i < len(voice_clips):
                    reconstructed_audio_list.append(_apply_fade(voice_clips[i], fade_type='both'))
        else:
            # 语音在前
            for i in range(max(len(voice_clips), len(silence_clips))):
                if i < len(voice_clips):
                    reconstructed_audio_list.append(_apply_fade(voice_clips[i], fade_type='both'))
                if i < len(silence_clips):
                    reconstructed_audio_list.append(_apply_fade(silence_clips[i], fade_type='out'))
                # 如果静音比语音多
                if i >= len(voice_clips) and i < len(silence_clips):
                    reconstructed_audio_list.append(_apply_fade(silence_clips[i], fade_type='out'))
    else:
        # 如果只有语音或只有静音
        if voice_clips:
            for clip in voice_clips:
                reconstructed_audio_list.append(_apply_fade(clip, fade_type='both'))
        if silence_clips:
            for clip in silence_clips:
                reconstructed_audio_list.append(_apply_fade(clip, fade_type='out'))
    
    # 拼接所有片段
    if not reconstructed_audio_list:
        # 如果没有片段，返回原始音频
        return audio_data
    
    return np.concatenate(reconstructed_audio_list)


def _apply_fade(audio_segment: np.ndarray, fade_type: str = 'both', fade_length: int = 512) -> np.ndarray:
    """
    对音频片段应用淡入淡出效果，以平滑拼接处
    
    Args:
        audio_segment: 音频片段
        fade_type: 淡入淡出类型 ('in', 'out', 'both')
        fade_length: 淡入淡出长度（样本数）
    
    Returns:
        应用淡入淡出后的音频片段
    """
    if len(audio_segment) <= 1:
        return audio_segment
        
    faded_segment = audio_segment.copy().astype(np.float32)
    fade_len = min(fade_length, len(faded_segment) // 2)
    
    if fade_type in ['in', 'both']:
        # 应用淡入效果
        fade_curve = np.linspace(0.0, 1.0, fade_len)
        faded_segment[:fade_len] = faded_segment[:fade_len] * fade_curve
    
    if fade_type in ['out', 'both']:
        # 应用淡出效果
        fade_curve = np.linspace(1.0, 0.0, fade_len)
        if fade_len <= len(faded_segment):
            faded_segment[-fade_len:] = faded_segment[-fade_len:] * fade_curve
    
    return faded_segment.astype(audio_segment.dtype)


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
            "duration": round(len(audio_data) / sample_rate, 3),
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


@app.post("/adjust_audio_length/")
async def adjust_audio_length_endpoint(
    audio_file: UploadFile = File(...), 
    target_duration: float = None
):
    """
    调整音频长度到指定时长
    接收音频文件和目标时长，返回调整后的音频文件
    """
    try:
        if target_duration is None:
            return JSONResponse(
                content={"error": "Please provide target_duration parameter"}, 
                status_code=400
            )
        
        # 读取上传的音频文件
        contents = await audio_file.read()
        
        # 使用librosa读取音频
        audio_data, sample_rate = librosa.load(io.BytesIO(contents), sr=16000)
        
        # 执行VAD检测获取人声段
        voice_segments = detect_voice_segments(audio_data, sample_rate)
        
        # 调整音频长度
        adjusted_audio = adjust_audio_length(
            audio_data, 
            sample_rate, 
            target_duration, 
            voice_segments
        )
        
        # 保存调整后的音频到内存中的字节流
        buffer = io.BytesIO()
        sf.write(buffer, adjusted_audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        # 返回调整后的音频文件
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=adjusted_{audio_file.filename}"
            }
        )
    
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