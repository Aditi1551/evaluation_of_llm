1. Install dependencies
```bash
pip install fastapi uvicorn pydantic[dotenv] python-multipart
pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cu121  # or cpu version
pip install transformers accelerate qwen-vl-utils pillow pymupdf
```

2. app.py – API server using Qwen2‑VL

3. Start the server
```bash
uvicorn app:app --reload --port 8080
```
