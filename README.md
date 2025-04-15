# Real-Time AI Voice Chat Web Application

## Overview
This is a real-time AI voice chat web application built with Python, FastAPI, and WebSockets. It uses Microsoft's DialoGPT-medium for conversation and gTTS for text-to-speech. The app supports Chinese (zh-TW) speech recognition and synthesis.

## Prerequisites
- Docker
- Docker Compose
- Modern web browser with WebSocket and Speech API support

## Setup
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. Create the static directory (if needed):
   ```bash
   mkdir static
   ```

3. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

4. Access the app at `http://localhost:8000`.

## Usage
- Click "Start Talking" to begin voice input.
- Speak into your microphone; the app will transcribe and display your text.
- The AI responds with text and synthesized speech, shown in the chat panel.
- Click "Stop Talking" to end the session.

## Notes
- Uses open-source DialoGPT-medium for free AI responses.
- gTTS provides free text-to-speech in Chinese.
- Ensure a stable internet connection for model downloads and WebSocket communication.
- Speech recognition relies on the browser's Web Speech API.