from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import speech_recognition as sr
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
    # Maintain conversation history
    chat_history_ids = None
    try:
        while True:
            data = await websocket.receive_json()
            user_text = data["text"]

            # Prepare input with conversation history
            new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
            if chat_history_ids is not None:
                bot_input_ids = new_user_input_ids
            else:
                bot_input_ids = new_user_input_ids

            # Generate AI response
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True, # If True, this'll make the model prevent picking the best answer
                top_k=50,
                top_p=0.95,
                temperature=0.5 # 0 is the most stable one, 1.5 is the most chaos one
            )

            # Decode AI response, excluding the user input
            ai_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            # Convert AI response to speech
            # init a gTTS object
            tts = gTTS(text=ai_response, lang='en')
            # init a temporary memory location 
            audio_buffer = io.BytesIO()
            # write the audio data into the audio buffer
            tts.write_to_fp(audio_buffer)
            # point the audio data to the begining
            audio_buffer.seek(0)
            # read data from the audio buffer, and encode it to base64, and decode it as json so that we can pass to front
            # sicne websocket don't know binary(MP3), but know JSON, we can convert it to base 64
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