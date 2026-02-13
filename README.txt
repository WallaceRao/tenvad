使用tenvad进行音频文件的VAD检测
利用fast API实现http服务
输入：
audio_file: 音频文件
输出：
json格式，分别为人声段开始和结束的时间戳
其他：
连续50ms检测到人声则认为开始讲话
连续50ms检测不到人声则认为开始静音