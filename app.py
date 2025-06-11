import os
import glob
import subprocess
import torch
import torchaudio
import streamlit as st
from yt_dlp import YoutubeDL
from speechbrain.pretrained import EncoderClassifier

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
        raise FileNotFoundError("No video file found after download.")

    if os.path.exists(output_path):
        os.remove(output_path)

    os.rename(temp_files[0], output_path)

def extract_audio():
    video_file = "video.mp4"
    if not os.path.exists(video_file):
        raise FileNotFoundError("video.mp4 not found.")

    cmd = ["ffmpeg", "-i", video_file, "-ar", "16000", "-ac", "1", "audio.wav", "-y", "-loglevel", "error"]
    subprocess.run(cmd, check=True)

def classify_english_accent(audio_path="audio.wav"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa",
        run_opts={"device": str(device)}
    )

    signal, _ = torchaudio.load(audio_path)
    signal = signal.to(device)

    out_prob, _, index, label = classifier.classify_batch(signal)
    accent = label[0]
    confidence = float(out_prob[0][index[0]]) * 100
    summary = f"The speaker probably has a **{accent}** accent (confidence: {confidence:.2f}%)."

    return accent, round(confidence, 2), summary

def analyze_accent_from_url(url):
    download_video(url)
    extract_audio()
    return classify_english_accent()

# ---- Streamlit UI ----
st.title("🎙️ English Accent Classifier")
st.write("Paste a YouTube video URL with clear speech and get an accent prediction.")

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
