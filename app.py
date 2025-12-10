import io
import fitz  # PyMuPDF
from typing import List, Any, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from PIL import Image

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ------------- Model load (Qwen2-VL) -------------

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # can use 2B for lighter load [web:105][web:119]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


# ------------- FastAPI app -------------

app = FastAPI(
    title="Invoice OCR with Qwen2-VL",
    version="1.0.0",
)


class InvoiceSchema(BaseModel):
    # you can extend this schema to match your needs
    invoice_number: str | None = None
    date_of_issue: str | None = None
    seller_info: Dict[str, Any] | None = None
    client_info: Dict[str, Any] | None = None
    invoice_items_table: Any | None = None
    invoice_summary_table: Any | None = None


# ------------- Helpers -------------

def pdf_to_images(pdf_bytes: bytes, dpi: int = 50) -> List[Image.Image]:
    """Render each page of PDF as PIL Image."""
    images: List[Image.Image] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    if not images:
        raise ValueError("No pages found in PDF")
    return images


def build_prompt(schema: dict) -> str:
    """Instruction for Qwen2-VL to return clean JSON."""
    return (
        "You are an expert invoice parser.\n"
        "Given the invoice image(s), extract the following fields as JSON:\n"
        f"{schema}\n\n"
        "Requirements:\n"
        "- Return ONLY valid, parsable JSON.\n"
        "- Do not include backticks or markdown.\n"
        "- If a field is missing, set it to null or an empty list.\n"
    )


def run_qwen2_vl_on_images(images: List[Image.Image], schema: dict) -> dict:
    # Qwen2-VL uses a messages structure with interleaved image + text. [web:105][web:119]
    message_content = []

    # add all pages as separate images
    for img in images:
        message_content.append({"type": "image", "image": img})

    # then add extraction instruction
    message_content.append({"type": "text", "text": build_prompt(schema)})

    messages = [{"role": "user", "content": message_content}]

    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # Try to parse JSON; if it fails, wrap as raw text
    import json

    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return {"raw_output": output_text}


# ------------- API endpoint -------------

@app.post("/api/invoice")
async def extract_invoice(
    file: UploadFile = File(..., description="Invoice PDF file"),
):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    pdf_bytes = await file.read()

    try:
        images = pdf_to_images(pdf_bytes, dpi=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF to image error: {e}")

    # simple schema dict – you can pass this from client if you want
    schema_dict = {
        "invoice_number": "string",
        "date_of_issue": "string",
        "seller_info": {
            "name": "string",
            "address": "string",
            "tax_id": "string",
        },
        "client_info": {
            "name": "string",
            "address": "string",
        },
        "invoice_items_table": "list of line items (description, quantity, unit_price, amount)",
        "invoice_summary_table": "totals, taxes, discounts, grand_total",
    }

    try:
        model_output = run_qwen2_vl_on_images(images, schema_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qwen2-VL inference error: {e}")

    return JSONResponse(
        content={
            "schema": schema_dict,
            "result": model_output,
        }
    )


@app.get("/health")
async def health():
    return {"status": "OK"}


# ------------- Entry point -------------

# Run with: uvicorn app:app --reload --port 8080
