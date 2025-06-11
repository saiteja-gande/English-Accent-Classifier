# 🎙️ English Accent Classifier

This is a simple and powerful web app that analyzes English from a **videos** and classifies the **regional English accent** (e.g., US, England, Australia, etc.) using **deep learning**.

Built using:
- 🧠 [SpeechBrain](https://speechbrain.readthedocs.io/) with a pretrained ECAPA-TDNN model  
- 🧰 `yt-dlp` to download audio from YouTube  
- 🎧 `ffmpeg` + `torchaudio` for audio processing  
- 🖥️ `Streamlit` for the web interface  

---

## 🚀 Live Demo

⚠️ **This app is not currently hosted online.**  
To try it out, follow the [Installation (Local)](#-installation-local) steps below and run it on your own machine.

---

## 📦 Features

- 🎥 Paste any **YouTube video URL**
- 🎧 Extracts clear mono audio at 16 kHz
- 🧠 Classifies English accent using a **pretrained SpeechBrain model**
- 💻 Automatically uses **CUDA (GPU)** if available
- 📊 Displays **accent** and **confidence score**

---

## 🔧 Installation (Local)

1. **Clone the repo**
   
  - `git clone https://github.com/your-username/english-accent-classifier.git`

  - `cd english-accent-classifier`

2. **Install the requirements**
   
  - `pip install -r requirements.txt`

4. **Run the App**
   
 - `streamlit run app.py`

