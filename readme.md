# ⚡ SLM Inference Engine

![Language](https://img.shields.io/badge/language-C++%20%7C%20Python-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Focus](https://img.shields.io/badge/focus-Systems%20Engineering-red?style=for-the-badge)


# Live Demo

[![Deployed on HuggingFace](https://img.shields.io/badge/Deployed%20On-HuggingFace-b7fc1d?style=for-the-badge&logo=huggingface&logoColor=black)](https://not-omega-inference.hf.space/)


A blazing-fast **50M parameter Small Language Model** inference engine built from scratch in C++, served via a Python FastAPI backend.

> **Benchmark:** ~28 tok/s | ~35ms/token on Intel i3 11th Gen (8GB RAM, Windows 11)

## 🧠 How to get the Model Weights (`model.bin`)

To train the model and generate your own `model.bin` and `tokenizer.bin` files, simply open the Google Colab notebook below. It is fully optimized to train on Colab's Free Tier GPU (T4).

1. Open the notebook link below.
2. Change the runtime to **GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
3. Run all cells to train and export the binary weights.


[https://colab.research.google.com/drive/1UEjL2YZmyxs5ZkdxN_aPT8ceApb3a-Xx?usp=sharing](https://colab.research.google.com/drive/1UEjL2YZmyxs5ZkdxN_aPT8ceApb3a-Xx?usp=sharing)

---

## 🏗️ Architecture

```
User / Browser
      │
      ▼
FastAPI Server (main.py)
      │  tiktoken tokenizer (encode prompt → token IDs)
      ▼
inference.exe  ◄── model.bin (GPT-2 style, 50M params)
      │  AVX2 SIMD + OpenMP parallelism
      ▼
Token IDs → FastAPI → tiktoken decode → JSON response
```

**Stack:**
- **Backend:** Python 3.12 + FastAPI + Uvicorn
- **Inference Engine:** C++17 with AVX2 SIMD + OpenMP (compiled to `inference.exe`)
- **Tokenizer:** tiktoken (GPT-2 encoding, 50,257 vocab)
- **Model:** Custom GPT-2-style binary format (`model.bin`)

---

## 🚀 Performance

| Metric | Value |
|---|---|
| Avg Request Latency | ~3556 ms / 100 tokens |
| Per Token Latency | ~35.57 ms/token |
| Throughput | **~28.12 tokens/sec** |
| Hardware | Intel i3-11th Gen, 8GB RAM |
| Platform | Windows 11 |

---

## 📁 Project Structure

```
INFERENCE ENGINE/
├── inference.cpp       # C++ inference engine (AVX2 + OpenMP)
├── inference.exe       # Compiled binary (Windows)
├── main.py             # FastAPI server
├── benchmark.py        # Performance benchmarking script
├── index.html          # Simple frontend UI
├── model.bin           # Model weights (binary format)
├── tokenizer.bin       # Tokenizer data
├── requirements.txt    # Python dependencies
├── SETUP_GUIDE.md      # Full setup instructions
└── .gitignore
```

---

## ⚙️ Quick Start

**1. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**3. Test it**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50}'
```

**4. Run benchmark**
```bash
python benchmark.py
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for full C++ compilation instructions and troubleshooting.

---

## 🔌 API Reference

### `GET /health`
Returns server status and file existence checks.

**Response:**
```json
{
  "status": "ok",
  "inference_exe_found": true,
  "model_bin_found": true,
  "working_directory": "C:/..."
}
```

### `POST /generate`
Generate text from a prompt.

**Request Body:**
```json
{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_k": 40
}
```

**Response:**
```json
{
  "prompt": "Once upon a time",
  "generated_text": "...generated output...",
  "tokens_in": 4,
  "tokens_out": 100,
  "latency_ms": 3556.65,
  "tokens_per_sec": 28.12
}
```

---

## 🔧 C++ Engine Features

- **AVX2 SIMD** — 8 floats processed per CPU instruction in matmul
- **FMA (Fused Multiply-Add)** — reduced rounding error + faster ops
- **OpenMP** — multi-threaded matrix multiplications, GELU, residuals
- **KV Cache** — avoids recomputing past token attention
- **Top-K Sampling** — efficient `partial_sort` for quality output

---

## 📦 Requirements

- Windows 10/11 (inference.exe is Windows binary)
- Python 3.10+
- 8GB RAM minimum
- CPU with AVX2 support (Intel 4th Gen+ / AMD Ryzen+)

---

## 📄 License

MIT License — free to use, modify, and distribute.
