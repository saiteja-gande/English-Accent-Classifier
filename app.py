import os
import glob
import subprocess
import torch
import torchaudio
import streamlit as st
import traceback
import uuid
from yt_dlp import YoutubeDL
from speechbrain.pretrained import EncoderClassifier
from collections import defaultdict

# --- Streamlit UI ---
st.title("🎙️ English Accent Classifier")
st.write("Paste a YouTube video URL with clear speech and get an accent prediction.")

# --- Load model once and cache it ---
@st.cache_resource
def load_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa",
        run_opts={"device": str(device)}
    )
    return classifier, device

# --- Clean temp files ---
def clean_temp_files():
    for f in glob.glob("temp_video.*"):
        try:
            os.remove(f)
        except Exception:
            pass
    for f in ["video.mp4", "audio.wav"]:
        if os.path.exists(f):
            os.remove(f)

# --- Download video (audio only) using cookies-from-browser ---
def download_video(url):
    unique_id = str(uuid.uuid4())
    temp_output = f"temp_video_{unique_id}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_output,
        'quiet': True,
        # Using cookies-from-browser for easier auth
        'cookies-from-browser': ('chrome',),  # You can change 'chrome' to 'firefox', 'edge', etc.
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    temp_files = glob.glob(f"temp_video_{unique_id}.*")
    if not temp_files:
        raise FileNotFoundError("No audio file downloaded.")

    if os.path.exists("video.mp4"):
        os.remove("video.mp4")
    os.rename(temp_files[0], "video.mp4")

# --- Extract mono 16kHz audio ---
def extract_audio(input_file="video.mp4", output_file="audio.wav"):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")
    cmd = [
        "ffmpeg", "-i", input_file,
        "-ar", "16000", "-ac", "1",
        output_file,
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

# --- Split audio into chunks ---
def split_audio(audio_path, chunk_seconds=30):
    signal, sr = torchaudio.load(audio_path)
    chunk_samples = chunk_seconds * sr
    chunks = []
    for start in range(0, signal.size(1), chunk_samples):
        end = min(start + chunk_samples, signal.size(1))
        chunks.append(signal[:, start:end])
    return chunks, sr

# --- Classify a single audio chunk ---
def classify_english_accent_from_tensor(signal, classifier, device):
    signal = signal.to(device)
    out_prob, _, index, label = classifier.classify_batch(signal)
    accent = label[0]
    confidence = float(out_prob[0][index[0]]) * 100
    return accent, confidence

# --- Classify full audio by chunks and aggregate results ---
def classify_full_audio(audio_path="audio.wav"):
    classifier, device = load_classifier()
    chunks, sr = split_audio(audio_path, chunk_seconds=30)

    results = []
    for chunk in chunks:
        if chunk.size(1) == 0:
            continue
        accent, confidence = classify_english_accent_from_tensor(chunk, classifier, device)
        results.append((accent, confidence))

    if not results:
        raise ValueError("No audio chunks to classify.")

    acc_dict = defaultdict(list)
    for a, c in results:
        acc_dict[a].append(c)

    avg_conf = {a: sum(cs)/len(cs) for a, cs in acc_dict.items()}
    best_accent = max(avg_conf, key=avg_conf.get)
    best_confidence = avg_conf[best_accent]

    summary = f"The speaker probably has a **{best_accent}** accent (average confidence: {best_confidence:.2f}%)."
    return best_accent, round(best_confidence, 2), summary

# --- Full pipeline ---
def analyze_accent_from_url(url):
    clean_temp_files()
    download_video(url)
    extract_audio()
    return classify_full_audio()

# --- User input ---
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
