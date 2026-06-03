# ASR Project - Klasifikasi Angka 0-9

PTU - Teknik Informatika - ITENAS Bandung

Project ini adalah sistem pengenalan suara (*Automatic Speech Recognition*) untuk mengklasifikasikan rekaman ucapan angka 0 sampai 9. Sistem menggunakan ekstraksi fitur MFCC dengan pendekatan *Temporal Chunking* yang dipadukan dengan model *Deep Learning* sederhana (Multi-Layer Perceptron / MLP).

## Gambaran Alur

```text
Dataset Audio (WAV)
  -> preprocess.py (Balancing & Feature Extraction)
  -> models/features.pkl
  -> train.py (Pelatihan Model MLP)
  -> models/asr_model.pkl + models/scaler.pkl
  -> app.py (Web Server Flask)
  -> Browser (Prediksi Suara Live)
```

## Alur Proses

1. **Persiapan Dataset & Balancing (`preprocess.py`)**
   - Dataset membaca folder kelas `0` hingga `9`.
   - Otomatis menyeimbangkan data menjadi **125 sampel per kelas**.
   - Kelas yang kurang di-*augmentasi* (Pitch Shift, Time Stretch, Noise).
   - Kelas yang lebih di-*undersample* (diambil acak).

2. **Preprocessing & Ekstraksi Fitur**
   - **Pembersihan:** *High-pass filter* (80Hz) untuk menghilangkan dengung (*hum*) ruangan, *Trim Silence* (30dB) untuk membuang keheningan di awal/akhir, dan *Pre-emphasis* untuk memperjelas konsonan.
   - **Fitur Akustik:** Menggunakan **MFCC (20)** beserta turunannya yaitu **Delta-MFCC (20)** untuk menangkap pergerakan dan transisi fonetik.
   - **Temporal Chunking:** Audio direpresentasikan secara sekuensial dengan membaginya menjadi 3 segmen waktu (awal, tengah, akhir). Setiap segmen dihitung nilai *Mean* dan *Std*-nya.
   - Total Dimensi Vektor: `(20 MFCC + 20 Delta) x 2 statistik x 3 segmen = 240 fitur`.

3. **Training Model (`train.py`)**
   - Fitur diskalakan menggunakan `StandardScaler`.
   - Menggunakan arsitektur jaringan saraf tiruan `MLPClassifier` dengan 2 *Hidden Layers* (256 dan 128 neuron).
   - Menggunakan metode optimasi Adam dan Early Stopping.
   - Model dievaluasi menggunakan *3-Fold Cross Validation* secara paralel (`n_jobs=-1`).
   - Hasil disimpan ke folder `models/runs/`.

4. **Inferensi Web (`app.py` & `main.js`)**
   - Antarmuka web menggunakan sistem **"Tekan & Tahan" (Hold to Speak)** agar durasi rekaman pas dengan ucapan, meminimalisir hening berlebih.
   - Mematikan filter bawaan browser (WebRTC *noise suppression & echo cancellation*) agar suara yang diproses 100% murni seperti data latih.
   - Menggunakan format WAV 16kHz Mono.
   - *Backend* Flask melakukan ekstraksi fitur (harus *exact match* dengan proses di `preprocess.py`).
   - Mengembalikan angka tebakan, skor kepercayaan (*confidence*), dan *Top 5 Probabilities*.

## Struktur Folder Utama

```text
asrp_app/
  app.py                 # Server Flask & API
  preprocess.py          # Script pembuatan dataset & fitur
  train.py               # Script pelatihan model MLP
  requirements.txt       # Dependencies
  dataset/               # Folder audio mentah (0-9)
  models/                # Model terlatih & scaler
    runs/                # Log evaluasi (CM, metrics, plot cv)
  static/js/main.js      # Logika frontend & WebRTC Audio
```

## Cara Menjalankan

### 1. Install dependency
```bash
pip install -r requirements.txt
```

### 2. Ekstraksi Fitur
Memproses seluruh dataset audio menjadi array fitur (*Temporal Chunking* 360 dimensi).
```bash
python preprocess.py
```

### 3. Training Model
Melatih model *Multi-Layer Perceptron* dan mencetak laporan klasifikasi.
```bash
python train.py
```

### 4. Menjalankan Web App
Menjalankan server web Flask untuk pengetesan.
```bash
python app.py
```
Buka browser dan akses: `http://localhost:5000`

## Stack Teknologi
- **Audio Processing:** `librosa`, `scipy`
- **Machine Learning:** `scikit-learn` (MLPClassifier, StandardScaler)
- **Web Backend:** `Flask`
- **Web Frontend:** HTML5, Vanilla JS (WebRTC / `MediaRecorder` API)
