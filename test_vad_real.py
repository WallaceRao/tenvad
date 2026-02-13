import requests
import json
import os

# 测试API
url = "http://localhost:8001/vad_detect/"

# 检查音频文件是否存在
audio_file_path = "cosyvoice3_result_dongbei.wav"
if not os.path.exists(audio_file_path):
    print(f"音频文件 {audio_file_path} 不存在")
    exit(1)

print(f"使用音频文件: {audio_file_path}")

# 准备要上传的音频文件
files = {"audio_file": open(audio_file_path, "rb")}

try:
    # 发送POST请求到VAD检测接口
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("VAD检测结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n检测到的人声段:")
        for i, segment in enumerate(result["voice_segments"]):
            print(f"  段 {i+1}: {segment['start']:.3f}s - {segment['end']:.3f}s, "
                  f"持续时间: {segment['end'] - segment['start']:.3f}s")
        
        print(f"\n总共检测到 {len(result['voice_segments'])} 个人声段")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("错误信息:", response.text)

except Exception as e:
    print(f"发生错误: {str(e)}")

finally:
    # 关闭文件
    files["audio_file"].close()