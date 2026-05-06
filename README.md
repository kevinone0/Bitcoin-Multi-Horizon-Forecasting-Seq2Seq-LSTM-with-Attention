# Bitcoin Multi-Horizon Forecasting: Seq2Seq LSTM with Attention

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.13-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Proyek ini berfokus pada pengembangan model *Deep Learning* tingkat lanjut untuk memprediksi harga penutupan (**Close Price**) Bitcoin. Dengan menggunakan pendekatan **Multivariate Multi-Horizon Forecasting**, model ini mampu memprediksi tren harga selama 24 jam ke depan berdasarkan data historis per jam dari tahun 2017 hingga 2023.

## 🚀 Fitur Utama
- **Multi-Step Forecasting**: Memprediksi 24 langkah waktu (*time steps*) sekaligus.
- **Multivariate Input**: Menggunakan 13 fitur teknis (RSI, MACD, ATR, Moving Averages, dll.) untuk meningkatkan akurasi.
- **Advanced Architecture**: Implementasi arsitektur **Seq2Seq LSTM** fungsional yang dilengkapi dengan **Multi-Head Attention**.
- **Custom Layers**: Menggunakan *layer* kustom untuk *Dropout*, *Normalization*, dan fungsi aktivasi guna optimasi model.

## 📊 Dataset & Feature Engineering
Dataset terdiri dari lebih dari **53.000 baris data per jam** (Hourly) Bitcoin.
- **Fitur Utama**: `Close`, `Volume USDT`, `RSI`, `MACD_Hist`, `ATR`, `KAMAO`.
- **Feature Engineering**: Penambahan fitur *Rolling Mean* (24h & 168h), *Standard Deviation*, *Price Change Percentage*, dan *Volatility*.
- **Preprocessing**: Normalisasi data menggunakan *MinMaxScaler* dan pembuatan jendela waktu (*Sliding Window*) sebesar 168 jam (7 hari) sebagai input.

## 🧠 Arsitektur Model
Proyek ini membandingkan beberapa arsitektur untuk mendapatkan hasil terbaik:
1. **Baseline LSTM**: Model LSTM standar sebagai pembanding performa.
2. **Seq2Seq LSTM (Functional)**: Menggunakan struktur *Encoder-Decoder* untuk menangkap dependensi jangka panjang.
3. **Attention Mechanism**: Integrasi *Multi-Head Attention* untuk memberikan bobot lebih pada variabel atau waktu yang lebih relevan dalam prediksi.

### Spesifikasi Teknis:
- **Input Shape**: `(batch, 168, 13)`
- **Output Shape**: `(batch, 24, 1)`
- **Optimizer**: Adam
- **Loss Function**: Custom Weighted Horizon Loss / MAE

## 🛠️ Instalasi
1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/bitcoin-forecasting-seq2seq-attention.git
   cd bitcoin-forecasting-seq2seq-attention
   ```
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Hasil & Evaluasi
Model ini berhasil mencapai stabilitas dalam prediksi jangka menengah (24 jam). 
*(Catatan: Anda dapat menyisipkan grafik loss atau plot prediksi hasil model Anda di sini untuk visualisasi yang lebih profesional).*

## 📂 Struktur Repositori
- `Rifqi_Zaghlul_Submission_Akhir_DLTM.ipynb`: Notebook utama berisi proses EDA hingga evaluasi.
- `baseline_lstm.keras`: File model tersimpan (Baseline).
- `seq2seq_functional.keras`: File model tersimpan (Advanced).
- `requirements.txt`: Daftar pustaka yang diperlukan.

## 👤 Penulis
**Rifqi Zaghlul**
- LinkedIn: [https://www.linkedin.com/in/rifqizaghlul]
- Email: [rifqizaghlul1@gmail.com]

---
