from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from llama_cpp import Llama
from bs4 import BeautifulSoup
import requests
import os
from pathlib import Path
import pytz

# ==== AI CONFIG ====
AI_NAME = "David AI v1"
AI_LOCATION = "Kolkata"
DEVELOPER = "David"
DEVELOPER_GITHUB = "https://github.com/david0154"
DEVELOPER_WEBSITE = "https://davidk.online/"
AI_WEBSITE = "https://ai.davidk.online/"
TIMEZONE = "Asia/Kolkata"

# ==== MODEL CONFIG ====
MODEL_FOLDER = "model"
MODEL_NAME = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/QuantFactory/TinyLlama-1.1B-Chat-GGUF/resolve/main/{MODEL_NAME}"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

# ==== AUTO DOWNLOAD MODEL IF MISSING ====
Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"⬇️ Downloading model: {MODEL_NAME}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        with requests.get(MODEL_URL, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Model downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        if not os.path.exists(MODEL_PATH):
            raise SystemExit("❌ Model download failed and file not found. Exiting...")

# ==== INIT FASTAPI ====
app = FastAPI()

# ==== CORS SETUP ====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Netlify domain if needed
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==== LOAD MODEL ====
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=2
)

# ==== UTILITIES ====
def get_current_ist():
    now = datetime.now(pytz.timezone(TIMEZONE))
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%I:%M %p")
    return f"📅 {date_str} | 🕒 {time_str} (IST)"

def get_indian_news():
    try:
        url = "https://www.hindustantimes.com/india-news"
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        headlines = soup.select("h3 a")
        news = [f"• {h.text.strip()}" for h in headlines[:3]]
        return "\n".join(news)
    except:
        return "⚠️ Failed to fetch news."

# ==== ROUTES ====

@app.get("/info")
def info():
    return {
        "ai_name": AI_NAME,
        "developer": DEVELOPER,
        "developer_github": DEVELOPER_GITHUB,
        "developer_website": DEVELOPER_WEBSITE,
        "ai_website": AI_WEBSITE,
        "location": AI_LOCATION,
        "timezone": TIMEZONE,
        "version": "1.0",
        "status": "online"
    }

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        print("🟡 Received:", message)

        if not message:
            return {"reply": "⚠️ Please type something."}

        # Commands
        if message.lower() == "!time":
            return {"reply": get_current_ist()}
        if message.lower() == "!news":
            return {"reply": get_indian_news()}

        # AI Reply
        prompt = f"[INST] {message} [/INST]"
        output = llm(prompt, max_tokens=200)
        print("📤 Model Output:", output)

        choices = output.get("choices", [])
        if not choices or not choices[0].get("text"):
            return {"reply": "⚠️ Model returned empty."}

        return {"reply": choices[0]["text"].strip()}

    except Exception as e:
        print("❌ /chat error:", e)
        return {"reply": f"⚠️ Internal error: {str(e)}"}
