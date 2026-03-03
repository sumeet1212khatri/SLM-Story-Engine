# main.py - SLM Inference Server
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import tiktoken
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40

# Tokenizer setup
try:
    enc = tiktoken.get_encoding("gpt2")
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Warning: tiktoken not found. Error: {e}")
    enc = None


@app.get("/health")
async def health_check():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path    = os.path.join(current_dir, "inference.exe")
    model_path  = os.path.join(current_dir, "model.bin")

    return {
        "status": "ok",
        "inference_exe_found": os.path.exists(exe_path),
        "model_bin_found":     os.path.exists(model_path),
        "working_directory":   current_dir
    }


@app.post("/generate")
async def generate_text(req: GenerateRequest):

    # 0. Tokenizer check
    if enc is None:
        raise HTTPException(
            status_code=500,
            detail="Tokenizer not loaded. Run: pip install tiktoken"
        )

    # 1. Encode prompt
    input_tokens = enc.encode(req.prompt)
    token_str    = ",".join(map(str, input_tokens))

    # 2. Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path    = os.path.join(current_dir, "inference.exe")
    model_path  = os.path.join(current_dir, "model.bin")

    print(f"DEBUG: exe   -> {exe_path}   exists={os.path.exists(exe_path)}")
    print(f"DEBUG: model -> {model_path} exists={os.path.exists(model_path)}")

    # 3. File existence checks
    if not os.path.exists(exe_path):
        raise HTTPException(
            status_code=500,
            detail=f"inference.exe nahi mili: {exe_path} — Pehle C++ compile karo!"
        )

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=500,
            detail=f"model.bin nahi mili: {model_path} — Model file same folder mein rakhni hai!"
        )

    # 4. Run C++ engine
    # FIX: temperature aur top_k ab subprocess ko pass ho rahe hain
    try:
        start_time = time.perf_counter()

        process = subprocess.run(
            [
                exe_path,
                token_str,
                str(req.max_tokens),
                str(req.temperature),   # <-- FIX: was missing before
                str(req.top_k),         # <-- FIX: was missing before
            ],
            capture_output=True,
            text=True,
            cwd=current_dir
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

    # 5. Error check
    if process.returncode != 0 and not process.stdout.strip():
        stdout_msg = process.stdout.strip() if process.stdout else ""
        stderr_msg = process.stderr.strip() if process.stderr else ""

        if "ERROR_MODEL_NOT_FOUND" in stdout_msg:
            raise HTTPException(status_code=500, detail="model.bin nahi mili! Same folder mein rakho.")
        elif "ERROR_ARGS" in stdout_msg:
            raise HTTPException(status_code=500, detail="C++ engine ko arguments galat mile.")
        else:
            raise HTTPException(
                status_code=500,
                detail=f"C++ Error | stdout: '{stdout_msg}' | stderr: '{stderr_msg}'"
            )

    # 6. Decode output token IDs
    try:
        output_str = process.stdout.strip()

        if not output_str:
            generated_ids = []
        else:
            generated_ids = []
            for x in output_str.split():
                try:
                    generated_ids.append(int(x))
                except ValueError:
                    print(f"DEBUG: skipping non-integer token: '{x}'")

        generated_text = enc.decode(generated_ids) if generated_ids else ""

        tokens_out     = len(generated_ids)
        tokens_per_sec = round(tokens_out / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0

        print(f"✅ Generated {tokens_out} tokens in {elapsed_ms:.2f}ms ({tokens_per_sec} tok/s)")

        return {
            "prompt":         req.prompt,
            "generated_text": generated_text,
            "tokens_in":      len(input_tokens),
            "tokens_out":     tokens_out,
            "latency_ms":     round(elapsed_ms, 2),
            "tokens_per_sec": tokens_per_sec
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding error: {str(e)}")