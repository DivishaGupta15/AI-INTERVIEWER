# avatar_main.py
# --------------------------------------------------------------
# Mic → faster-whisper (CPU)  → GPT-4o-mini (stream) → ElevenLabs TTS (stream)
#        (ASR)                     (LLM)                   (audio chunks)
# --------------------------------------------------------------
import asyncio, os, time, sounddevice as sd, numpy as np, tempfile, sys
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from elevenlabs import ElevenLabs
# --------------------------------------------------------------
# ▶ CONFIG
FS            = 16_000              # sample-rate
BLOCK         = 1024                # mic read size (64 ms)
PAUSE_MS      = 600                 # silence ≥0.6s ⇒ end of turn
WHISPER_SIZE  = "small"             # tiny / small / base …
VAD_THRESH    = 1500                # tweak if you get false positives
# --------------------------------------------------------------
# ▶ QUEUES
llm_queue  : asyncio.Queue[str] = asyncio.Queue()  # ASR ➜ GPT
tts_queue  : asyncio.Queue[str] = asyncio.Queue()  # GPT ➜ TTS
audio_q    : asyncio.Queue[bytes] = asyncio.Queue()# TTS ➜ player
# --------------------------------------------------------------
# ▶ ASR  (1-second rolling buffer, CPU int8)
whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

async def mic_loop():
    buf = np.empty(0, dtype=np.int16)
    last_voice = time.time()

    with sd.RawInputStream(samplerate=FS, blocksize=BLOCK, dtype="int16") as mic:
        while True:
            block, _ = mic.read(BLOCK)
            pcm      = np.frombuffer(block, dtype=np.int16)
            buf      = np.concatenate([buf, pcm])[-FS:]       # keep ≤1 s

            if np.abs(pcm).mean() > VAD_THRESH:               # simple VAD
                last_voice = time.time()

            if time.time() - last_voice > PAUSE_MS/1000:
                if buf.any():
                    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    write(wav, FS, buf)
                    segs, _ = whisper_model.transcribe(wav, beam_size=1)
                    os.remove(wav)

                    text = "".join(s.text for s in segs).strip()
                    if text:
                        print(f"\n✅ {text}")
                        await llm_queue.put(text)
                buf = np.empty(0, dtype=np.int16)
                last_voice = time.time()
# --------------------------------------------------------------
# ▶ GPT  (token stream)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY not set"))

async def llm_loop():
    while True:
        prompt = await llm_queue.get()
        reply  = ""
        stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[
                {"role":"system","content":"You are a friendly AI avatar interviewer."},
                {"role":"user",   "content":prompt},
            ],
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                reply += delta
                print(delta, end="", flush=True)          # live console
                await tts_queue.put(delta)                # hand tokens to TTS
        await tts_queue.put("<eos>")                      # mark done
# --------------------------------------------------------------
# ▶ TTS  (ElevenLabs stream → bytes → audio_q)
el_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY") or sys.exit("ELEVEN_API_KEY not set"))
VOICE_ID  = "JBFqnCBsd6RMkjVDRZzb"   # pick any

async def tts_loop():
    sentence = ""
    while True:
        token = await tts_queue.get()
        if token == "<eos>":                           # flush sentence
            if sentence.strip():
                async for chunk in el_client.text_to_speech.stream(
                        voice_id=VOICE_ID,
                        text=sentence,
                        model_id="eleven_multilingual_v2",
                        output_format="pcm_16000"):
                    await audio_q.put(chunk)
            sentence = ""
        else:
            sentence += token
# --------------------------------------------------------------
# ▶ Local speaker playback
async def audio_player():
    with sd.RawOutputStream(samplerate=FS, blocksize=2048,
                            channels=1, dtype='int16') as spk:
        while True:
            data = await audio_q.get()
            spk.write(data)
# --------------------------------------------------------------
# ▶ MAIN
async def main():
    await asyncio.gather(
        mic_loop(),        # mic → llm_queue
        llm_loop(),        # llm_queue → tts_queue
        tts_loop(),        # tts_queue → audio_q
        audio_player(),    # audio_q  → speakers
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
