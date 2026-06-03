# MATERI PRESENTASI & DOKUMENTASI: PENGENALAN UCAPAN ANGKA 0-9

Dokumen ini disusun untuk menjelaskan secara runut dan jelas mengenai metodologi, proses pengolahan dataset, hingga alur kerja sistem (dari input sampai output). Sangat cocok digunakan sebagai acuan poin-poin presentasi (PPT) atau laporan.

---

## 1. METODE YANG DIGUNAKAN

Sistem ini menggabungkan teknik Pemrosesan Sinyal Digital (DSP) dengan algoritma Jaringan Saraf Tiruan (*Deep Learning*).

*   **Metode Ekstraksi Fitur (Pemrosesan Suara):** Menggunakan **MFCC** (*Mel-Frequency Cepstral Coefficients*) dan **Delta-MFCC**. MFCC berfungsi meniru cara telinga manusia membedakan bentuk suara, sedangkan Delta-MFCC melacak *kecepatan pergerakan/transisi* antar suku kata (misal: transisi dari huruf 'L' ke 'i' pada kata "Lima").
*   **Pendekatan Sekuensial:** Menggunakan metode **Temporal Pooling**. Daripada mengambil rata-rata suara secara keseluruhan (yang merusak urutan kata), suara dibagi menjadi 3 segmen waktu (Awal, Tengah, Akhir).
*   **Metode Klasifikasi (Machine Learning):** Menggunakan arsitektur **Multi-Layer Perceptron (MLP) Classifier**. Ini adalah *Feedforward Neural Network* dengan 2 *hidden layer* (256 neuron dan 128 neuron) yang terbukti sangat andal dalam memetakan pola numerik yang kompleks.

---

## 2. BAGAIMANA DATASET DIOLAH (DARI PREPROCESSING HINGGA TRAINING)

Sebelum model bisa menebak suara, model harus dilatih dari dataset mentah. Proses pelatihan dilakukan melalui dua script utama: `preprocess.py` dan `train.py`.

### Tahap A: Persiapan Dataset & Augmentasi (`preprocess.py`)
Sistem membaca folder dataset yang berisi rekaman WAV mentah dari angka 0-9. Agar model kebal terhadap berbagai jenis mikrofon dan gaya bicara, dilakukan *Data Augmentation* (Augmentasi):
*   **Pitch Shift:** Mengubah nada suara (lebih tinggi/rendah).
*   **Time Stretch:** Mengubah tempo (bicara lebih cepat/lambat).
*   **Heavy Noise Injection:** Menyuntikkan suara desis/bising ruangan buatan. Ini adalah langkah krusial agar model nantinya tidak kebingungan saat pengguna merekam di tempat bising.
*   Data diseimbangkan hingga masing-masing angka memiliki porsi yang rata.

### Tahap B: Pembersihan Sinyal (*Audio Cleaning*)
Setiap file audio yang masuk akan "dicuci" agar murni:
1.  **High-pass Filter (80Hz):** Membuang frekuensi sangat rendah seperti dengung (*hum*) ruangan atau hembusan angin.
2.  **Trim Silence (30dB):** Memotong/membuang bagian hening di awal dan akhir rekaman.
3.  **Pre-emphasis:** Sebuah filter yang secara spesifik *memperjelas dan menajamkan suara konsonan* (seperti desis huruf 'S' pada "Satu" atau 'T' pada "Tiga").
4.  **Normalize:** Meratakan volume semua rekaman agar seragam.

### Tahap C: Ekstraksi 240 Fitur (Vektorisasi)
Suara yang sudah bersih kemudian dipotong menjadi **3 Segmen Waktu** (Awal, Tengah, Akhir). 
Untuk setiap segmen, sistem menghitung rata-rata (*Mean*) dan sebaran (*Std*) dari 20 MFCC dan 20 Delta-MFCC.
*   **Perhitungan:** `(20 MFCC + 20 Delta) x 2 statistik x 3 segmen = 240 Fitur`.
Audio kini telah direpresentasikan dalam bentuk deretan angka matematis (Vektor 1D berjumlah 240) yang padat informasi fonetik.

### Tahap D: Pelatihan Model (`train.py`)
1.  **Standard Scaler:** Ke-240 fitur tersebut distandarisasi skalanya agar neural network mudah mencerna data.
2.  **Pelatihan MLP:** Data disuapkan ke dalam *MLP Classifier*. Menggunakan fungsi aktivasi *ReLU*, optimasi *Adam*, dan fitur *Early Stopping*.
3.  **Cross-Validation:** Evaluasi dilakukan menggunakan metode pemisahan *GroupShuffleSplit* untuk mencegah *data leakage*, sehingga akurasi tes (berkisar ~95%) benar-benar merepresentasikan keandalan di dunia nyata.

---

## 3. CARA KERJA SISTEM (DARI INPUT HINGGA OUTPUT DI WEB)

Ketika *user* membuka aplikasi web (`app.py`), proses *live inference* (deteksi langsung) terjadi dalam hitungan milidetik dengan urutan sebagai berikut:

**Langkah 1: Input Audio (Sisi *Frontend / Browser*)**
*   Pengguna menggunakan antarmuka **"Tekan & Tahan" (Hold to Speak)**. Desain ini memastikan durasi perekaman *persis* sepanjang ucapan, sehingga tidak ada hening/bising tambahan yang ikut terekam.
*   Fitur pemrosesan bawaan browser (WebRTC *noise suppression & echo cancellation*) secara sadar **dinonaktifkan** via JavaScript agar suara yang direkam benar-benar mentah (*raw*) dan tidak terdistorsi, menyamai kualitas data latih.
*   Audio mentah ini dikirim ke server (*Backend*).

**Langkah 2: Pemrosesan Kesamaan (*Backend*)**
*   Begitu file mendarat di server Flask, sistem melakukan tahap *Audio Cleaning* yang **sama persis** dengan saat *training* (High-pass, Trim, Pre-emphasis, Normalize).

**Langkah 3: Transformasi ke Fitur**
*   Suara pengguna dikonversi menjadi MFCC + Delta-MFCC, dibagi 3 segmen, lalu dibentuk menjadi **240 Fitur**.
*   240 fitur ini dilewatkan ke dalam *Scaler* yang sudah disimpan saat pelatihan (`scaler.pkl`).

**Langkah 4: Prediksi & Output**
*   Fitur yang telah distandarisasi dimasukkan ke dalam otak model utama (`asr_model.pkl`).
*   Model (MLP) membaca urutan waktu dan karakteristik fitur, lalu menghitung persentase kemiripan dengan angka 0 hingga 9.
*   **Output:** Model mengembalikan tebakan final beserta persentase *confidence* (Top 5 Probabilitas). Angka ini langsung ditampilkan di layar web pengguna bersama visualisasi 3D MFCC.
