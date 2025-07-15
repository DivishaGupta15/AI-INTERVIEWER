import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
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



openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["ELEVEN_API_KEY"] = os.getenv("ELEVEN_API_KEY")


whisper_model = whisper.load_model("small")  

def record_audio(
        filename="user_input.wav",
        fs=44100,
        silence_thresh=0.015,    #silence threshold 
        silence_duration=2.0,    #silence detection time
        max_record=60            
    ):
   
    print("Speak now… (Ctrl-C to end interview)")

    buf_len = int(fs * silence_duration)
    ring = np.zeros(buf_len, dtype=np.float32)       

    frames = []                 
    start = time.time()

    try:
        with sd.InputStream(samplerate=fs, channels=1) as stream:
            while True:
                block, _ = stream.read(int(fs * 0.1))      
                block = block.flatten()
                frames.append(block)

                ring = np.roll(ring, -len(block))
                ring[-len(block):] = block
                rms = np.sqrt(np.mean(ring**2))

                # this function makes it stop if long enough silence
                if rms < silence_thresh and (time.time() - start) > 0.3:  #for pause in the beginning it ignores that
                    
                    silent_for = 0
                    while rms < silence_thresh:
                        silent_for += 0.1
                        if silent_for >= silence_duration:
                            raise StopIteration
                        block, _ = stream.read(int(fs * 0.1))
                        block = block.flatten()
                        frames.append(block)
                        ring = np.roll(ring, -len(block))
                        ring[-len(block):] = block
                        rms = np.sqrt(np.mean(ring**2))

                if (time.time() - start) >= max_record:
                    print("Max recording length reached.")
                    break

    except StopIteration:
        pass    

    audio_data = np.concatenate(frames)
    write(filename, fs, audio_data)
    return filename

def transcribe_audio(filename):
    result = whisper_model.transcribe(filename)
    print(f"You said: {result['text']}")
    return result["text"]

def get_ai_response(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a calm and professional job interviewer."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def speak(text):
    print(f"\n{text}\n")
    audio = generate(
        text=text,
        voice=Voice(
            voice_id="UgBBYS2sOqTuMpoF3BR0",
            settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
        ),
        model="eleven_multilingual_v2"
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

if __name__ == "__main__":
    run_interview()
