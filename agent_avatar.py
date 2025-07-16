# agent_avatar.py
import os, time, tempfile, warnings, wave
from pathlib import Path

import numpy as np
import whisper
import sounddevice as sd
from scipy.io.wavfile import write

import os
import openai 
from elevenlabs import generate, play, Voice, VoiceSettings 

from run_avatar import animate_avatar  

import time 
import numpy as np
import subprocess 
import streamlit as st

# SadTalker loader (make sure sadtalker_loader.py is correctly configured)
from sadtalker_loader import sad_talker

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ── API KEYS & MODELS ─────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY") or st.stop("OPENAI_API_KEY not set")
os.environ["ELEVEN_API_KEY"] = os.getenv("ELEVEN_API_KEY", "")

whisper_model = whisper.load_model("small")          # CPU model is fine
AVATAR_IMG    = Path(__file__).resolve().parent / "assets" / "avatar.png"

# ── MIC RECORDING ────────────────────────────────────────────────
def record_audio(
    filename="user_input.wav",
    fs=16_000,
    silence_thresh=0.008,
    silence_duration=1.2,
    max_record=60,
    device=None,
):
    """Record mic until the speaker pauses, write a WAV, return its path or None."""
    ring_len = int(fs * silence_duration)
    ring     = np.zeros(ring_len, dtype=np.float32)
    frames, heard = [], False
    t0 = time.time()

    try:
        with sd.InputStream(samplerate=fs, channels=1, device=device) as stream:
            while True:
                blk, _ = stream.read(int(fs * 0.1))
                blk = blk.flatten()
                ring = np.roll(ring, -len(blk))
                ring[-len(blk):] = blk

                if np.abs(ring).mean() > silence_thresh:
                    heard = True
                    frames.append(blk)
                    last_voice = time.time()

                if heard and (time.time() - last_voice) > silence_duration:
                    break
                if heard and (time.time() - t0) > max_record:
                    break
                if not heard and (time.time() - t0) > 10:
                    return None
    except KeyboardInterrupt:
        return None

    if not frames:
        return None

    write(filename, fs, np.concatenate(frames))
    return filename

# ── ASR, GPT, TTS ────────────────────────────────────────────────
def transcribe_audio(wav: str) -> str:
    return whisper_model.transcribe(wav)["text"].strip()

def ai_reply(prompt: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a calm and professional job interviewer."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

def tts_to_wav(text: str, wav_path: str):
    """Generate speech with ElevenLabs and save as 24 kHz mono WAV."""
    stream = generate(
        text=text,
        voice=Voice(
            voice_id="UgBBYS2sOqTuMpoF3BR0",
            settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
        ),
        model="eleven_multilingual_v2",
    )
    pcm_parts = [
        (chunk if isinstance(chunk, (bytes, bytearray))
         else np.asarray(chunk, dtype=np.int16).tobytes())
        for chunk in stream
    ]
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(24_000)
        wf.writeframes(b"".join(pcm_parts))

# ── SadTalker wrapper ────────────────────────────────────────────
def wav_to_mp4(wav_path: str) -> str:
    """Run SadTalker and return the real MP4 path it produces."""
    video_path = sad_talker.test(
        str(AVATAR_IMG),     # source image
        str(wav_path),       # driven audio
        result_dir="results"
    )

    play(audio)
    
def animate_avatar(audio_path, image_path="avatar.png", output_path="results/latest_animation.mp4"):
    print("Animating avatar...")
    sadtalker_cmd = [
        "python", "SadTalker/inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", "results",
        "--enhancer", "gfpgan",
        "--preprocess", "full",
        "--still",  # Static head pose
        "--output_name", "latest_animation"
    ]
    subprocess.run(sadtalker_cmd, cwd="SadTalker")
    print(f"Animation saved to {output_path}")
    return output_path

def run_interview():
    print("Welcome to the AI Interviewer. Press Ctrl+C to quit.\n")
    while True:
        try:
            audio_file = record_audio()
            user_input = transcribe_audio(audio_file)
            ai_reply = get_ai_response(user_input)
            speak(ai_reply)
        except KeyboardInterrupt:
            print("\nInterview ended.")
            break
        except Exception as e:
            print(f"Error: {e}")
            
    return video_path

# ── Streamlit helpers ────────────────────────────────────────────
def speak(text: str):
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    tts_to_wav(text, wav_path)

    video_path = wav_to_mp4(wav_path)

    with AVATAR_IMG.open("rb") as f:
        st.image(f.read(), width=250)

    with open(video_path, "rb") as f:
        st.video(f.read(), format="video/mp4")

    with open(wav_path, "rb") as f:
        st.audio(f.read(), format="audio/wav")

    st.markdown(f"**Interviewer:** {text}")


# ── Streamlit front-end when run directly ───────────────────────
if __name__ == "__main__":
    st.title("🎤 Live Job-Interview Avatar")
    if st.button("Record a question"):
        wav_in = record_audio()
        if wav_in:
            you = transcribe_audio(wav_in)
            st.markdown(f"**You:** {you}")
            speak(ai_reply(you))
        else:
            st.warning("No speech detected.")

# backward compatibility for app.py
get_ai_response = ai_reply
