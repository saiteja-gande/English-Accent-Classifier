# 🎙️ English Accent Classifier

This is a simple and powerful web app that analyzes spoken English from **YouTube videos** and classifies the **regional English accent** (e.g., US, England, Australia, etc.) using **deep learning**.

Built using:
- 🧠 [SpeechBrain](https://speechbrain.readthedocs.io/) with a pretrained ECAPA-TDNN model  
- 🧰 `yt-dlp` (uses browser cookies) to download audio from YouTube  
- 🎧 `ffmpeg` + `torchaudio` for robust audio processing  
- 🖥️ `Streamlit` for the web interface  

---

## 🚀 Live Demo

🌐 Try it here: [APP](https://saiteja-gande-english-accent-classifier.streamlit.app/)

> ⚠️ If YouTube-based downloads doesn't work properly. See [Using Cookies](#-using-cookies-for-youtube-downloads-optional-but-recommended).

---

## 📦 Features

- 🎥 Accepts **YouTube and other video URLs (such as loom)**
- 🔐 Supports **age-restricted / authenticated videos** via `--cookies-from-browser`
- 🔊 Extracts high-quality mono audio at 16 kHz
- 🧠 Classifies English accents using a **pretrained SpeechBrain model**
- 💻 Automatically uses **CUDA (GPU)** if available
- 📊 Displays **predicted accent** and **confidence score**
- 🔄 Processes full-length audio for **greater accuracy**

---

## 🔧 Installation (Local)

1. **Clone the repo**
   
  - `git clone https://github.com/your-username/english-accent-classifier.git`

  - `cd english-accent-classifier`

2. **Install the requirements**
   
  - `pip install -r requirements.txt`

4. **Run the App**
   
 - `streamlit run app.py`

## 🙏 Credits

This project was made possible thanks to the following open-source tools and communities:

- [SpeechBrain](https://speechbrain.readthedocs.io/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [FFmpeg](https://ffmpeg.org/)
- [torchaudio](https://pytorch.org/audio/stable/)
- [Streamlit](https://streamlit.io/)


