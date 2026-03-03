import requests
import time
import statistics
import psutil
import os

# Server URL
URL = "http://localhost:8000/generate"

# Benchmark Config
RUNS = 20        # Kitni baar test karega
WARMUP = 3       # Pehli baar slow hota hai, usko ignore karega
MAX_TOKENS = 100 # Har baar kitne tokens generate karega

payload = {
    "prompt": "Once upon a time there was a curious child who",
    "max_tokens": MAX_TOKENS,
    "temperature": 0.8,
    "top_k": 40
}

latencies = []
process = psutil.Process(os.getpid())

print("\n🔥 Running warmup (Server ko garam kar rahe hain)...")
for _ in range(WARMUP):
    try:
        requests.post(URL, json=payload)
        print(".", end="", flush=True)
    except Exception as e:
        print(f"\nError contacting server: {e}")
        exit()

print("\n⚡ Running benchmark (Asli Test shuru)...")

cpu_before = psutil.cpu_percent(interval=None)
mem_before = psutil.virtual_memory().used

for i in range(RUNS):
    start = time.perf_counter()
    response = requests.post(URL, json=payload)
    elapsed = (time.perf_counter() - start) * 1000 # Convert to ms
    latencies.append(elapsed)
    
    # Calculate tokens generated in this run
    try:
        data = response.json()
        tokens_out = data.get("tokens_out", MAX_TOKENS) # Default to max if key missing
    except:
        tokens_out = MAX_TOKENS

    print(f"Run {i+1}/{RUNS}: {elapsed:.2f}ms ({tokens_out} tokens)")

cpu_after = psutil.cpu_percent(interval=None)
mem_after = psutil.virtual_memory().used

avg_latency = sum(latencies) / len(latencies)
std_latency = statistics.stdev(latencies)

# Calculations
ms_per_token = avg_latency / MAX_TOKENS
tokens_per_sec = 1000 / ms_per_token

print("\n================ 🚀 CPU BENCHMARK RESULTS ================")
print(f"Runs: {RUNS}")
print(f"Average Request Latency: {avg_latency:.2f} ms")
print(f"Per Token Latency: {ms_per_token:.2f} ms/token (Lower is Better)")
print(f"Throughput: {tokens_per_sec:.2f} tokens/sec (Higher is Better)")
print(f"CPU Usage Change: {cpu_after - cpu_before:.1f}%")
print("==========================================================")