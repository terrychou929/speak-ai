# main.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import speech_recognition as sr
import numpy as np
import base64
import io
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load open-source model (DialoGPT)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.get("/")
async def get():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_text = data["text"]

            # Generate AI response
            inputs = tokenizer(user_text, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], max_length=100, pad_token_id=tokenizer.eos_token_id)
            ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Convert AI response to speech
            tts = gTTS(text=ai_response, lang='zh-tw')
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

            # Send back user text, AI response, and audio
            await websocket.send_json({
                "user": user_text,
                "ai": ai_response,
                "audio": audio_base64
            })
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()