# 🎙️ PodForge — Free AI Podcast Generator

Generate podcasts from scripts using **Microsoft Edge TTS voices** — completely free, no API keys needed.

## Features

- 🎤 **50+ natural-sounding voices** (Microsoft Edge Neural TTS)
- 🌍 Multiple accents: US, UK, Australian, Indian, Spanish, French, German
- ⚡ **Preview voices** before generating
- 📝 Script templates: Interview, Debate, Panel, Solo
- 🎛️ Adjustable speed & pause timing
- 📥 **Download as MP3**
- 💰 **100% free** — no API keys, no subscriptions

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

You also need **ffmpeg** for audio stitching:

- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html and add to PATH

### 2. Run the server

```bash
python server.py
```

### 3. Open in browser

Go to **http://localhost:5000**

## How to Use

1. **Paste your script** — Format each line as `Speaker Name: dialogue text`
2. **Click "Detect Speakers"** — The app finds all unique speakers
3. **Assign voices** — Pick from 50+ Microsoft Neural voices, preview each one
4. **Adjust settings** — Title, speed, pause length
5. **Generate** — Hit the button, wait for it to process
6. **Download** — Get your podcast as an MP3 file

## Script Format

```
Host: Welcome to the show everyone!
Guest: Thanks for having me.
Host: So tell us about your work.
Guest: Well, it all started when...
```

Also supports bracket format:
```
[Host] Welcome to the show!
[Guest] Thanks for having me.
```

## Hosting (Optional)

The app runs locally by default. If you want to host it:

- **Render.com** — Free tier, 750 hours/month
- **HuggingFace Spaces** — Free, great for ML apps
- **Fly.io** — Free tier with 3 shared VMs
- **Your own VPS** — Run with gunicorn

## Tech Stack

- **Backend:** Python, Flask, edge-tts, pydub
- **Frontend:** Vanilla HTML/CSS/JS (no build step)
- **TTS Engine:** Microsoft Edge Neural TTS (free, no key required)

## License

MIT — do whatever you want with it.
