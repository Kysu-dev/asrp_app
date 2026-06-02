# ASR Project - Klasifikasi Angka 0-9

PTU - Teknik Informatika - ITENAS Bandung

Project ini adalah sistem pengenalan suara untuk mengklasifikasikan rekaman angka 0 sampai 9. Alurnya dimulai dari penggabungan dataset beberapa speaker, preprocessing audio menjadi fitur MFCC, training model klasifikasi, lalu inferensi melalui web app Flask.

## Gambaran Alur

```text
Dataset speaker
  -> preprocessdataset.py
  -> dataset/0-9
  -> preprocess.py
  -> models/features.pkl
  -> train.py
  -> models/asr_model.pkl + models/scaler.pkl
  -> app.py
  -> Web prediksi suara
```

## Alur Proses

1. Merge dataset
   - Dataset dari beberapa folder speaker digabung ke folder `dataset/`.
   - Nama file dinormalisasi dan dipetakan ke kelas angka 0-9.

2. Preprocessing audio
   - Audio dibaca dengan `librosa`.
   - Noise reduction dilakukan pada tahap preprocessing dataset.
   - Silence di-trim, audio dinormalisasi, lalu dipotong atau dipad ke durasi tetap.
   - Fitur yang diambil adalah MFCC, delta, dan delta-delta.
   - Feature vector dinormalisasi per-sample sebelum disimpan.

3. Training model
   - Data fitur dibagi menjadi train dan test.
   - Fitur diskalakan lagi dengan `StandardScaler`.
   - Model klasifikasi dilatih menggunakan `SVC`.
   - Model final disimpan ke `models/asr_model.pkl`.
   - Scaler final disimpan ke `models/scaler.pkl`.
   - Report evaluasi disimpan ke folder `models/runs/<timestamp>/`.

4. Inferensi web
   - User merekam suara dari mikrofon selama 2 detik.
   - Frontend mengubah rekaman browser menjadi WAV sebelum upload.
   - Audio dikirim ke endpoint Flask `/api/predict`.
   - Backend melakukan preprocessing dan ekstraksi fitur yang konsisten dengan training.
   - Model mengembalikan prediksi angka, confidence, dan top-5 kelas.

## Struktur Folder

```text
asrp_app/
  app.py
  preprocess.py
  preprocessdataset.py
  train.py
  requirements.txt
  dataset/
    0/
    1/
    ...
  models/
    asr_model.pkl
    scaler.pkl
    features.pkl
    runs/
      <timestamp>/
  static/
    css/style.css
    js/main.js
  templates/
    index.html
```

## Cara Menjalankan

### 1. Install dependency

```bash
pip install -r requirements.txt
```

### 2. Siapkan dataset mentah

Taruh rekaman audio per speaker ke folder dataset mentah, lalu jalankan:

```bash
python preprocessdataset.py
```

### 3. Jalankan preprocessing fitur

```bash
python preprocess.py
```

Output:

- `models/features.pkl`

### 4. Training model

```bash
python train.py
```

Output utama:

- `models/asr_model.pkl`
- `models/scaler.pkl`
- `models/runs/<timestamp>/classification_report.txt`
- `models/runs/<timestamp>/confusion_matrix.png`
- `models/runs/<timestamp>/f1_per_class.png`
- `models/runs/<timestamp>/cv_scores.png`

### 5. Jalankan web app

```bash
python app.py
```

Buka:

- `http://localhost:5000`

## Fitur Web App

- Rekam suara langsung dari mikrofon selama 2 detik.
- Konversi rekaman browser ke WAV sebelum dikirim ke backend.
- Kirim audio ke Flask untuk diprediksi.
- Tampilkan angka hasil prediksi.
- Tampilkan confidence dan top-5 kelas teratas.
- Endpoint status menampilkan apakah model benar-benar berhasil dimuat.

## Detail Preprocessing

- Sample rate: `16000 Hz`
- Durasi target audio: `2 detik`
- MFCC: `20 koefisien`
- Turunan fitur: `delta` dan `delta-delta`
- Statistik fitur: `mean` dan `std`
- Normalisasi feature vector: `(x - mean) / (std + 1e-8)`
- Scaling model: `StandardScaler`

Setiap sampel audio diubah menjadi vektor fitur fixed-length berukuran 120:

```text
20 MFCC x 3 jenis fitur x 2 statistik = 120 fitur
```

## Model Yang Dipakai

- Algoritma: SVM (`scikit-learn`)
- Kernel: `rbf`
- Class weight: `balanced`
- Probabilistic output: aktif, supaya bisa menampilkan confidence dan top-5 prediksi

## Endpoint API

### POST `/api/predict`

Upload audio lalu sistem mengembalikan hasil prediksi.

Form data:

- `audio`: file audio WAV

Contoh response:

```json
{
  "success": true,
  "prediction": "5",
  "confidence": 87.34,
  "top5": [
    {"label": "5", "probability": 87.34},
    {"label": "3", "probability": 7.12},
    {"label": "8", "probability": 2.44}
  ]
}
```

### GET `/api/status`

Memeriksa apakah model dan scaler sudah berhasil dimuat.

Response contoh:

```json
{
  "model_loaded": true,
  "error": null
}
```

## Ringkasan Singkat

1. Gabungkan dataset speaker ke `dataset/`.
2. Jalankan `preprocess.py` untuk membuat fitur.
3. Jalankan `train.py` untuk melatih model dan scaler final.
4. Jalankan `app.py` untuk membuka web prediksi suara.
