import requests
import json
import os
from pathlib import Path

# 测试API
base_url = "http://localhost:8061"

# 检查音频文件是否存在
audio_file_path = "9.wav"
if not os.path.exists(audio_file_path):
    print(f"音频文件 {audio_file_path} 不存在")
    exit(1)

print(f"使用音频文件: {audio_file_path}")

# 首先获取音频的基本信息
print("\n1. 测试VAD检测功能...")
vad_url = f"{base_url}/vad_detect/"
files = {"audio_file": open(audio_file_path, "rb")}

try:
    response = requests.post(vad_url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("VAD检测结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        original_duration = result["duration"]
        print(f"\n原始音频时长: {original_duration}秒")
        
        # 关闭并重新打开文件用于下一个请求
        files["audio_file"].close()
        
        print(f"\n2. 测试音频长度调整功能...")
        print(f"将音频长度调整为 {original_duration/2:.2f} 秒 (缩短一半)")
        
        # 测试音频缩短功能
        adjust_url = f"{base_url}/adjust_audio_length/"
        files = {"audio_file": open(audio_file_path, "rb")}
        params = {"target_duration": original_duration/2}
        
        response = requests.post(adjust_url, files=files, params=params)
        
        if response.status_code == 200:
            # 保存调整后的音频文件
            output_filename = f"adjusted_{Path(audio_file_path).stem}_half.wav"
            with open(output_filename, "wb") as f:
                f.write(response.content)
            print(f"音频已成功调整并保存为: {output_filename}")
            
            # 检查文件大小
            file_size = os.path.getsize(output_filename)
            print(f"调整后文件大小: {file_size} 字节")
        else:
            print(f"音频调整失败，状态码: {response.status_code}")
            print("错误信息:", response.text)
        
        print(f"\n3. 测试音频长度调整功能...")
        print(f"将音频长度调整为 {original_duration*1.5:.2f} 秒 (延长一半)")
        
        # 关闭并重新打开文件用于下一个请求
        files["audio_file"].close()
        files = {"audio_file": open(audio_file_path, "rb")}
        params = {"target_duration": original_duration*1.5}
        
        response = requests.post(adjust_url, files=files, params=params)
        
        if response.status_code == 200:
            # 保存调整后的音频文件
            output_filename = f"adjusted_{Path(audio_file_path).stem}_longer.wav"
            with open(output_filename, "wb") as f:
                f.write(response.content)
            print(f"音频已成功调整并保存为: {output_filename}")
            
            # 检查文件大小
            file_size = os.path.getsize(output_filename)
            print(f"调整后文件大小: {file_size} 字节")
        else:
            print(f"音频调整失败，状态码: {response.status_code}")
            print("错误信息:", response.text)
            
    else:
        print(f"VAD检测失败，状态码: {response.status_code}")
        print("错误信息:", response.text)

except Exception as e:
    print(f"发生错误: {str(e)}")

finally:
    # 确保文件被关闭
    try:
        files["audio_file"].close()
    except:
        pass