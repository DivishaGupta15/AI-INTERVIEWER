import subprocess
import os

def animate_avatar(audio_file="sample.wav", image_file="avatar.png"):
    sadtalker_dir = "SadTalker"  
    command = [
        "python", "inference.py",
        "--driven_audio", f"../{audio_file}",
        "--source_image", f"../{image_file}",
        "--result_dir", "../results",
        "--preprocess", "full",
        "--enhancer", "gfpgan"
    ]
    print("Starting avatar animation with SadTalker...")
    subprocess.run(command, cwd=sadtalker_dir)
    print("Animation complete. Check the 'results' folder.")
