# VAD音频检测API

使用TenVAD进行音频文件的VAD检测，利用FastAPI实现HTTP服务。

## 功能特性

- 输入：音频文件
- 输出：JSON格式，包含人声段开始和结束的时间戳
- 规则：连续50ms检测到人声则认为开始讲话，连续50ms检测不到人声则认为开始静音

## 快速开始

### 启动服务

```bash
conda activate cosyvoice
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API接口

- `POST /vad_detect/`: 上传音频文件进行VAD检测
- `GET /`: 获取API状态信息
- `GET /docs`: 查看交互式API文档

### 使用示例

通过浏览器访问 `http://localhost:8000/docs` 可以查看交互式API文档，并进行在线测试。

## 文件说明

- `main.py`: 主应用程序文件，包含FastAPI服务和VAD检测逻辑
- `test_api.py`: API测试脚本
- `requirements.txt`: 项目依赖列表
- `README.txt`: 项目原始需求文档

## 技术实现

- 使用TenVAD库进行语音活动检测
- 实现了符合需求的50ms连续检测规则
- 支持多种常见音频格式（通过librosa库）
- 返回标准化的JSON格式结果