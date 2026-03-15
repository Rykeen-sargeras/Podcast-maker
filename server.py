"""
PodForge — Free AI Podcast Generator
Backend: Flask + edge-tts (Microsoft's free TTS engine)
Run: python server.py
Then open: http://localhost:5000
"""

import asyncio
import os
import re
import json
import uuid
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import edge_tts
from pydub import AudioSegment
from werkzeug.utils import secure_filename

from script_generator import (
    ingest_all_sources,
    generate_script,
    get_youtube_transcript,
    scrape_article,
    STYLE_PROMPTS,
)

app = Flask(__name__, static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── Popular edge-tts voices (curated for podcast quality) ───────────────────
VOICES = {
    "English (US)": [
        {"id": "en-US-GuyNeural", "name": "Guy", "gender": "Male", "accent": "US"},
        {"id": "en-US-JennyNeural", "name": "Jenny", "gender": "Female", "accent": "US"},
        {"id": "en-US-AriaNeural", "name": "Aria", "gender": "Female", "accent": "US"},
        {"id": "en-US-DavisNeural", "name": "Davis", "gender": "Male", "accent": "US"},
        {"id": "en-US-AmberNeural", "name": "Amber", "gender": "Female", "accent": "US"},
        {"id": "en-US-AndrewNeural", "name": "Andrew", "gender": "Male", "accent": "US"},
        {"id": "en-US-BrandonNeural", "name": "Brandon", "gender": "Male", "accent": "US"},
        {"id": "en-US-ChristopherNeural", "name": "Christopher", "gender": "Male", "accent": "US"},
        {"id": "en-US-CoraNeural", "name": "Cora", "gender": "Female", "accent": "US"},
        {"id": "en-US-ElizabethNeural", "name": "Elizabeth", "gender": "Female", "accent": "US"},
        {"id": "en-US-EricNeural", "name": "Eric", "gender": "Male", "accent": "US"},
        {"id": "en-US-JacobNeural", "name": "Jacob", "gender": "Male", "accent": "US"},
        {"id": "en-US-MichelleNeural", "name": "Michelle", "gender": "Female", "accent": "US"},
        {"id": "en-US-MonicaNeural", "name": "Monica", "gender": "Female", "accent": "US"},
        {"id": "en-US-RogerNeural", "name": "Roger", "gender": "Male", "accent": "US"},
        {"id": "en-US-SteffanNeural", "name": "Steffan", "gender": "Male", "accent": "US"},
    ],
    "English (UK)": [
        {"id": "en-GB-RyanNeural", "name": "Ryan", "gender": "Male", "accent": "UK"},
        {"id": "en-GB-SoniaNeural", "name": "Sonia", "gender": "Female", "accent": "UK"},
        {"id": "en-GB-ThomasNeural", "name": "Thomas", "gender": "Male", "accent": "UK"},
        {"id": "en-GB-LibbyNeural", "name": "Libby", "gender": "Female", "accent": "UK"},
        {"id": "en-GB-MaisieNeural", "name": "Maisie", "gender": "Female", "accent": "UK"},
    ],
    "English (Australia)": [
        {"id": "en-AU-NatashaNeural", "name": "Natasha", "gender": "Female", "accent": "AU"},
        {"id": "en-AU-WilliamNeural", "name": "William", "gender": "Male", "accent": "AU"},
    ],
    "English (India)": [
        {"id": "en-IN-NeerjaNeural", "name": "Neerja", "gender": "Female", "accent": "IN"},
        {"id": "en-IN-PrabhatNeural", "name": "Prabhat", "gender": "Male", "accent": "IN"},
    ],
    "Spanish": [
        {"id": "es-MX-DaliaNeural", "name": "Dalia", "gender": "Female", "accent": "MX"},
        {"id": "es-MX-JorgeNeural", "name": "Jorge", "gender": "Male", "accent": "MX"},
        {"id": "es-ES-ElviraNeural", "name": "Elvira", "gender": "Female", "accent": "ES"},
        {"id": "es-ES-AlvaroNeural", "name": "Alvaro", "gender": "Male", "accent": "ES"},
    ],
    "French": [
        {"id": "fr-FR-DeniseNeural", "name": "Denise", "gender": "Female", "accent": "FR"},
        {"id": "fr-FR-HenriNeural", "name": "Henri", "gender": "Male", "accent": "FR"},
    ],
    "German": [
        {"id": "de-DE-KatjaNeural", "name": "Katja", "gender": "Female", "accent": "DE"},
        {"id": "de-DE-ConradNeural", "name": "Conrad", "gender": "Male", "accent": "DE"},
    ],
}


def parse_script(script_text):
    """
    Parse script in format:
        Speaker Name: Their dialogue here.
        Another Speaker: Their dialogue.

    Also supports:
        [Speaker Name] Their dialogue here.
        SPEAKER NAME: Their dialogue.
    """
    lines = []
    speakers = set()

    for raw_line in script_text.strip().split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Try "Speaker: text" format
        match = re.match(r"^([A-Za-z0-9_ .'-]+?):\s*(.+)$", raw_line)
        if not match:
            # Try "[Speaker] text" format
            match = re.match(r"^\[([A-Za-z0-9_ .'-]+?)\]\s*(.+)$", raw_line)

        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            speakers.add(speaker)
            lines.append({"speaker": speaker, "text": text})
        else:
            # If no speaker detected, append to previous line or treat as narration
            if lines:
                lines[-1]["text"] += " " + raw_line
            else:
                speakers.add("Narrator")
                lines.append({"speaker": "Narrator", "text": raw_line})

    return lines, sorted(speakers)


async def generate_audio_segment(text, voice_id, output_path, rate="+0%", pitch="+0Hz"):
    """Generate a single audio segment using edge-tts."""
    communicate = edge_tts.Communicate(text, voice_id, rate=rate, pitch=pitch)
    await communicate.save(output_path)


def combine_audio_files(file_list, output_path, pause_ms=400):
    """Combine multiple audio files with pauses between them."""
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=pause_ms)

    for f in file_list:
        try:
            segment = AudioSegment.from_file(f)
            combined += segment + pause
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
            continue

    combined.export(output_path, format="mp3", bitrate="192k")
    return output_path


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/voices", methods=["GET"])
def get_voices():
    return jsonify(VOICES)


@app.route("/api/parse", methods=["POST"])
def parse():
    """Parse a script and return detected speakers."""
    data = request.json
    script = data.get("script", "")
    lines, speakers = parse_script(script)
    return jsonify({"lines": lines, "speakers": speakers})


@app.route("/api/preview", methods=["POST"])
def preview_voice():
    """Generate a short preview of a voice."""
    data = request.json
    voice_id = data.get("voice_id", "en-US-GuyNeural")
    text = data.get("text", "Hey everyone, welcome to the podcast. Let's dive right in!")

    preview_id = f"preview_{uuid.uuid4().hex[:8]}"
    preview_path = OUTPUT_DIR / f"{preview_id}.mp3"

    asyncio.run(generate_audio_segment(text, voice_id, str(preview_path)))

    return jsonify({"audio_url": f"/audio/{preview_id}.mp3"})


@app.route("/api/generate", methods=["POST"])
def generate_podcast():
    """Generate the full podcast audio from script + voice assignments."""
    data = request.json
    script = data.get("script", "")
    voice_map = data.get("voice_map", {})  # {speaker_name: voice_id}
    rate = data.get("rate", "+0%")
    pause_between = data.get("pause_ms", 400)
    podcast_title = data.get("title", "podcast")

    lines, speakers = parse_script(script)

    if not lines:
        return jsonify({"error": "No dialogue lines found in script."}), 400

    # Default voice assignment for unmapped speakers
    default_voices = ["en-US-GuyNeural", "en-US-JennyNeural", "en-US-DavisNeural",
                      "en-US-AriaNeural", "en-US-BrandonNeural", "en-US-CoraNeural"]
    for i, speaker in enumerate(speakers):
        if speaker not in voice_map:
            voice_map[speaker] = default_voices[i % len(default_voices)]

    # Generate each line
    session_id = uuid.uuid4().hex[:12]
    segment_files = []

    for idx, line in enumerate(lines):
        seg_path = OUTPUT_DIR / f"{session_id}_seg{idx:04d}.mp3"
        voice_id = voice_map.get(line["speaker"], "en-US-GuyNeural")

        try:
            asyncio.run(generate_audio_segment(line["text"], voice_id, str(seg_path), rate=rate))
            segment_files.append(str(seg_path))
        except Exception as e:
            print(f"Error generating segment {idx}: {e}")
            continue

    if not segment_files:
        return jsonify({"error": "Failed to generate any audio segments."}), 500

    # Combine all segments
    safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", podcast_title)[:50]
    final_path = OUTPUT_DIR / f"{safe_title}_{session_id}.mp3"
    combine_audio_files(segment_files, str(final_path), pause_ms=pause_between)

    # Cleanup segment files
    for f in segment_files:
        try:
            os.remove(f)
        except:
            pass

    file_size = os.path.getsize(str(final_path))
    duration_approx = file_size / (192000 / 8)  # rough estimate from bitrate

    return jsonify({
        "audio_url": f"/audio/{final_path.name}",
        "filename": final_path.name,
        "duration_seconds": round(duration_approx, 1),
        "segments_generated": len(segment_files),
    })


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    return send_from_directory("output", filename)


@app.route("/api/cleanup", methods=["POST"])
def cleanup():
    """Remove old generated files (older than 1 hour)."""
    cutoff = time.time() - 3600
    removed = 0
    for f in OUTPUT_DIR.glob("*.mp3"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    for f in UPLOAD_DIR.glob("*"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    return jsonify({"removed": removed})


# ─── Script Generator API Routes ────────────────────────────────────────────

@app.route("/api/styles", methods=["GET"])
def get_styles():
    """Return available podcast styles."""
    styles = {}
    for key, info in STYLE_PROMPTS.items():
        styles[key] = {
            "name": key.replace("_", " ").title(),
            "speakers": info["speakers"],
            "description": info["description"],
        }
    return jsonify(styles)


@app.route("/api/ingest/youtube", methods=["POST"])
def ingest_youtube():
    """Fetch transcript from a YouTube video."""
    data = request.json
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    result = get_youtube_transcript(url)
    return jsonify(result)


@app.route("/api/ingest/article", methods=["POST"])
def ingest_article():
    """Scrape text from a web article."""
    data = request.json
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    result = scrape_article(url)
    return jsonify(result)


@app.route("/api/ingest/pdf", methods=["POST"])
def ingest_pdf():
    """Upload and extract text from a PDF."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
    file.save(str(save_path))

    from script_generator import extract_pdf_text
    result = extract_pdf_text(str(save_path))
    return jsonify(result)


@app.route("/api/generate-script", methods=["POST"])
def generate_script_route():
    """Generate a podcast script from sources using selected AI provider."""
    data = request.json

    provider = data.get("provider", "gemini").strip().lower()
    api_key = data.get("api_key", "").strip()

    # Check env vars as fallback based on provider
    if not api_key:
        env_map = {
            "gemini": "PF_GEMINI_KEY",
            "openrouter": "PF_OPENROUTER_KEY",
            "groq": "PF_GROQ_KEY",
        }
        api_key = os.environ.get(env_map.get(provider, ""), "")

    if not api_key:
        help_urls = {
            "gemini": "https://aistudio.google.com/apikey",
            "openrouter": "https://openrouter.ai/keys",
            "groq": "https://console.groq.com/keys",
        }
        return jsonify({"error": f"API key required for {provider}. Get one free at {help_urls.get(provider, '')}"}), 400

    topic = data.get("topic", "")
    style = data.get("style", "interview")
    duration_min = int(data.get("duration_min", 5))
    duration_max = int(data.get("duration_max", 15))
    custom_speakers = data.get("custom_speakers", None)
    instructions = data.get("additional_instructions", "")

    # Collect sources that were already ingested client-side
    sources = data.get("sources", [])

    # Also handle any inline text/notes
    raw_texts = data.get("raw_texts", [])
    if raw_texts:
        for txt in raw_texts:
            if isinstance(txt, str) and txt.strip():
                sources.append({
                    "title": "User Notes",
                    "text": txt,
                    "source": "user input",
                    "error": None,
                })

    if not sources and not topic:
        return jsonify({"error": "Provide at least a topic or some source material."}), 400

    result = generate_script(
        api_key=api_key,
        sources=sources,
        style=style,
        duration_min=duration_min,
        duration_max=duration_max,
        custom_speakers=custom_speakers,
        additional_instructions=instructions,
        topic=topic,
        provider=provider,
    )

    return jsonify(result)


if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  🎙️  PodForge — Free AI Podcast Generator")
    print("=" * 56)
    print("  Open in your browser: http://localhost:5000")
    print("=" * 56 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
