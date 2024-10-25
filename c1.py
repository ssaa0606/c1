from fastapi.responses import FileResponse
import os
import sys
import argparse
import logging
import time
import random
import torchaudio
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# 跨域设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 初始化 CosyVoice 模型
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=True, load_onnx=False, fp16=True)

@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    try:
        # 生成音频数据
        logging.info(f"Received request for text: {tts_text} with speaker id: {spk_id}")
        model_output = cosyvoice.inference_sft(tts_text, spk_id, stream=False)

        # 生成带时间戳和随机数的文件名
        timestamp = int(time.time())  # 返回秒级时间戳
        random_number = random.randint(1000, 9999)  # 生成4位随机数
        output_dir = "/root/CosyVoice/"
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory checked or created: {output_dir}")

        audio_file_path = f"{output_dir}{timestamp}_{random_number}.wav"  # 使用f-string嵌入路径

        # 使用 torchaudio 保存 WAV 文件
        for i, j in enumerate(model_output):
            # 每个生成的块保存为 wav 文件（假设每次生成一个完整的文件）
            torchaudio.save(audio_file_path, j['tts_speech'], 22050)  # 22050 是采样率

        logging.info(f"WAV file saved successfully: {audio_file_path}")

        # 返回完整的音频文件
        return FileResponse(audio_file_path, media_type="audio/wav", filename=f"{timestamp}_{random_number}.wav")

    except Exception as e:
        logging.error(f"Error occurred while processing the request: {e}")
        return {"error": "An error occurred while generating the audio file."}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
