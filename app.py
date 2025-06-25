import os
import glob
import subprocess
import torch
import torchaudio
import streamlit as st
import traceback
from yt_dlp import YoutubeDL
from speechbrain.pretrained import EncoderClassifier

# ---- Streamlit UI ----
st.title("🎙️ English Accent Classifier")
st.write("Paste a YouTube video URL with clear speech and get an accent prediction.")

# --- Load model (optional: enable caching) ---
# Uncomment @st.cache_resource if you're confident caching works correctly
# @st.cache_resource
def load_classifier():
    device = torch.device("cpu")  # Use "cuda" for GPU
    classifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa",
        run_opts={"device": str(device)}
    )
    return classifier, device

# --- Download best audio (no postprocessing) ---
def download_video(url, output_path="video.mp4"):
    temp_output = "temp_video.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_output,
        'quiet': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    temp_files = glob.glob("temp_video.*")
    if not temp_files:
        raise FileNotFoundError("No audio file found after download.")

    if os.path.exists(output_path):
        os.remove(output_path)

    os.rename(temp_files[0], output_path)

# --- Convert to mono WAV @ 16kHz ---
def extract_audio(input_file="video.mp4", output_file="audio.wav"):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")

    cmd = [
        "ffmpeg", "-i", input_file,
        "-ar", "16000", "-ac", "1", output_file,
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

# --- Accent Classification ---
def classify_english_accent(audio_path="audio.wav"):
    classifier, device = load_classifier()

    signal, sr = torchaudio.load(audio_path)
    signal = signal.to(device)

    if signal.shape[1] == 0:
        raise ValueError("Audio file appears to be empty or silent.")

    out_prob, _, index, label = classifier.classify_batch(signal)
    accent = label[0]
    confidence = float(out_prob[0][index[0]]) * 100
    summary = f"The speaker probably has a **{accent}** accent (confidence: {confidence:.2f}%)."

    return accent, round(confidence, 2), summary

# --- Full Analysis Pipeline ---
def analyze_accent_from_url(url):
    try:
        download_video(url)
        extract_audio()
        accent, score, summary = classify_english_accent()
        return accent, score, summary
    finally:
        # Clean up temporary files
        for f in ["video.mp4", "audio.wav"]:
            if os.path.exists(f):
                os.remove(f)

# --- User Input ---
video_url = st.text_input("Enter YouTube Video URL")

if st.button("Analyze Accent"):
    if not video_url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Downloading and processing..."):
            try:
                accent, score, summary = analyze_accent_from_url(video_url)
                st.success(summary)
                st.metric(label="Detected Accent", value=accent)
                st.metric(label="Confidence Score", value=f"{score}%")
            except Exception as e:
                st.error(f"Error: {e}")
                st.text(traceback.format_exc())
