# make_voice.py
from pathlib import Path
from TTS.api import TTS

# -- 1. load model (cached after first run) -------------------
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

# -- 2. sentence to speak -------------------------------------
text = "Hello, how are you? Welcome to the interview."

# -- 3. output path -------------------------------------------
out_dir  = Path("audio")
out_dir.mkdir(exist_ok=True)
wav_path = out_dir / "interviewer.wav"

# -- 4. synthesize --------------------------------------------
tts.tts_to_file(text=text, file_path=wav_path)
print("✅  Saved ->", wav_path.resolve())
