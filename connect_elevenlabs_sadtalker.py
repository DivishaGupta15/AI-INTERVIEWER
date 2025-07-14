"""
Run inside the repo venv:
    python connect_elevenlabs_sadtalker.py
"""

import os, sys, subprocess, tempfile, shutil
from pathlib import Path

from elevenlabs import generate, save, Voice, VoiceSettings

# paths
ROOT   = Path(__file__).parent
ASSETS = ROOT / "assets"
OUT    = ROOT / "out"
OUT.mkdir(exist_ok=True)

text_path  = ASSETS / "script.txt" # script file which will tell the sadtalker what to say in the video
image_path = ASSETS / "neutral2.jpg" #make sure to change this if you use another neutral shot
wav_path   = OUT   / "interview.wav" # voice
video_path = OUT   / "talking.mp4" # output

# use a 100 % ASCII temp work-dir for SadTalker
WORK = Path(tempfile.gettempdir()) / "sadtalker_work"
WORK.mkdir(exist_ok=True)

# copy portrait to a plain-ASCII location too
temp_image = WORK / "src.jpg"
try:
    shutil.copy(str(image_path), str(temp_image))   # fast path
except OSError:                                     # fallback for odd Win bugs
    temp_image.write_bytes(image_path.read_bytes()) # byte-for-byte copy

#ElevenLabs TTS
api_key = os.getenv("ELEVEN_API_KEY")
if not api_key:
    sys.exit("ELEVEN_API_KEY not set (PowerShell →  setx ELEVEN_API_KEY \"sk-…\")")

script = text_path.read_text().strip()
print(f" Text ({len(script)} chars):", script[:80] + "…")

audio = generate(
    api_key = api_key,
    text    = script,
    model   = "eleven_multilingual_v2",
    voice   = Voice(
        voice_id = "TxGEqnHWrfWFTfGW9XjX",          #voice ID, change this as needed if you need another type of voice
        settings = VoiceSettings(stability=0.38, similarity_boost=0.90)
    )
)
save(audio, wav_path)
print("✔ Saved TTS  →", wav_path)

#SadTalker inference
cmd = [
    sys.executable, "inference.py",
    "--source_image", str(temp_image),
    "--driven_audio", str(wav_path),
    "--checkpoint_dir", "checkpoints",
    "--result_dir",     str(WORK),
    "--enhancer",       "gfpgan",
    "--preprocess",     "full"
]
print("▶ Running SadTalker …")
subprocess.run(cmd, check=True)

# move the first (only) mp4 back to ./out as talking.mp4
generated = next(WORK.glob("*.mp4"))
shutil.move(generated, video_path)
print("✅ DONE — see →", video_path.resolve())

