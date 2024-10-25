from fastapi.responses import FileResponse
import os
import sys
import argparse
import logging
import time
import random
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    # 生成音频数据
    model_output = cosyvoice.inference_sft(tts_text, spk_id)

    # 生成带时间戳和随机数的文件名
    timestamp = int(time.time())  # 返回秒级时间戳
    random_number = random.randint(1000, 9999)  # 生成4位随机数
    audio_file_path = f"/root/CosyVoice/{timestamp}_{random_number}.wav"  # 使用f-string嵌入路径

    # 将生成的音频保存为 WAV 文件
    with open(audio_file_path, 'wb') as f:
        for i in model_output:
            tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            f.write(tts_audio)

    # 返回完整的音频文件
    return FileResponse(audio_file_path, media_type="audio/wav", filename=f"{timestamp}_{random_number}.wav")


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
