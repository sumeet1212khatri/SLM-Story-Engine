# Setup Guide — SLM Inference Engine

Full step-by-step guide to compile, configure, and run the SLM Inference Engine on Windows 11.

---

## Prerequisites

| Tool | Version | Download |
|---|---|---|
| Python | 3.10+ | https://python.org |
| MinGW-w64 (GCC) | 13.0+ | https://winlibs.com |
| Git | Any | https://git-scm.com |

> **Important:** You need a CPU with **AVX2 support** (Intel 4th Gen+ or AMD Ryzen+).

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/slm-inference-engine.git
cd slm-inference-engine
```

---

## Step 2: Set Up Python Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Verify tiktoken is working:
```bash
python -c "import tiktoken; enc = tiktoken.get_encoding('gpt2'); print('tiktoken OK')"
```

---

## Step 3: Compile the C++ Inference Engine

Open Command Prompt or Git Bash in the project folder:

```bash
g++ -O3 -march=native -fopenmp -mavx2 -mfma -std=c++17 inference.cpp -o inference.exe -lm
```

### Flag Explanations

| Flag | Purpose |
|---|---|
| `-O3` | Maximum compiler optimization |
| `-march=native` | Auto-detect and use your CPU's best instructions |
| `-fopenmp` | Enable multi-threading with OpenMP |
| `-mavx2` | Enable AVX2 SIMD (256-bit vector ops) |
| `-mfma` | Enable Fused Multiply-Add instructions |
| `-std=c++17` | Use C++17 standard |

Verify compilation:
```bash
inference.exe
# Expected output: ERROR_ARGS  (means it compiled correctly!)
```

---

## Step 4: Add Model File

Place your `model.bin` file in the **same folder** as `inference.exe`:

```
INFERENCE ENGINE/
+-- inference.exe
+-- model.bin        <- must be here
+-- main.py
+-- ...
```

> `model.bin` format: 5 x int32 config header (n_layer, n_head, n_embd, block_size, vocab_size) then all float32 weights packed sequentially.

---

## Step 5: Start the Server

```bash
venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at: http://localhost:8000

Health check:
```bash
curl http://localhost:8000/health
```

---

## Step 6: Test Generation

```bash
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d "{\"prompt\": \"The future of AI is\", \"max_tokens\": 50}"
```

Or open `index.html` in your browser for the UI.

---

## Step 7: Run Benchmark

```bash
python benchmark.py
```

Expected on i3 11th Gen: ~28 tokens/sec, ~35 ms/token.

---

## Troubleshooting

**inference.exe nahi mili** — Compile karo Step 3 se. Same folder mein honi chahiye.

**model.bin nahi mili** — model.bin ko main.py wale folder mein rakho.

**tiktoken not found** — `pip install tiktoken`

**AVX2 errors during compilation** — Try without AVX2 flags:
```bash
g++ -O3 -fopenmp -std=c++17 inference.cpp -o inference.exe -lm
```

**Linux compile command:**
```bash
g++ -O3 -march=native -fopenmp -mavx2 -mfma -std=c++17 inference.cpp -o inference -lm
```

---

## API Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Input text |
| `max_tokens` | int | 100 | Max tokens to generate |
| `temperature` | float | 0.8 | Randomness (0.1=focused, 1.5=creative) |
| `top_k` | int | 40 | Top-K sampling pool size |