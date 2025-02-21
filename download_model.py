import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from huggingface_hub import snapshot_download

# Create the required cache directories if they don't exist
os.makedirs("cache/whisper", exist_ok=True)
os.makedirs("cache/diarization", exist_ok=True)

model_id = "KBLab/kb-whisper-large"

print("Downloading Whisper model (KBLab/kb-whisper-large)...")
# Download the model into the cache/whisper directory
whisper_dir = snapshot_download(repo_id=model_id, cache_dir="cache/whisper")
print(f"Whisper model downloaded to: {whisper_dir}")

print("Downloading Diarization model (pyannote/speaker-diarization-3.1)...")
# Download the diarization model into the cache/diarization directory
diarization_dir = snapshot_download(repo_id="pyannote/speaker-diarization-3.1", cache_dir="cache/diarization")
print(f"Diarization model downloaded to: {diarization_dir}")
