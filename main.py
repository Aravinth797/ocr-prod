import os
import re
import json
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
from starlette.formparsers import MultiPartParser

app = FastAPI()

UPLOAD_DIR = "uploads"
RESULTS_FILE = os.path.join(UPLOAD_DIR, "results.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# allow big batch uploads
orig_init = MultiPartParser.__init__
def patched_init(self, *args, **kwargs):
    kwargs["max_files"] = 5000
    orig_init(self, *args, **kwargs)
MultiPartParser.__init__ = patched_init

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def preprocess_image(fp):
    try:
        img = Image.open(fp).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.MedianFilter())
        img.save(fp)
    except Exception as e:
        print(f"Preprocess err on {fp}: {e}")

def extract_amount_from_ocr(ocr_res):
    try:
        for line in ocr_res[0]:
            txt = line[1][0]
            if re.search(r"\d", txt):
                return re.sub(r"[^\d.]", "", txt)
        return "Not Detected"
    except:
        return "OCR Error"

def process_file(fp: str, fn: str):
    # --- 1) get expected from raw ---
    try:
        raw_ocr = ocr.ocr(fp, cls=True)
        expected = extract_amount_from_ocr(raw_ocr)
    except:
        expected = None

    # --- 2) preprocess + re‑OCR for “extracted” ---
    preprocess_image(fp)
    try:
        proc_ocr = ocr.ocr(fp, cls=True)
        extracted = extract_amount_from_ocr(proc_ocr)
    except Exception as e:
        extracted = f"OCR Error: {e}"

    # --- 3) compare cleaned digits ---
    clean_exp = re.sub(r"[^\d]", "", expected or "") 
    clean_ext = re.sub(r"[^\d]", "", extracted or "")
    is_match = (clean_exp == clean_ext) and bool(clean_exp)

    # --- 4) stash in results.json ---
    rec = {
        "filename": fn,
        "expected_amount": expected,
        "amount": extracted,
        "match": is_match,
        "preview_url": f"/uploads/{fn}"
    }
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data = [d for d in data if d["filename"] != fn]
                data.append(rec)
                f.seek(0); f.truncate(); json.dump(data, f, indent=2)
        else:
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump([rec], f, indent=2)
    except Exception as e:
        print(f"Error saving {fn} → {e}")

@app.post("/upload")
async def upload_images(bg: BackgroundTasks, files: list[UploadFile] = File(...)):
    resp = []
    for file in files:
        safe = re.sub(r'[\\/]', '_', file.filename)
        path = os.path.join(UPLOAD_DIR, safe)
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        bg.add_task(process_file, path, safe)
        resp.append({"filename": safe, "status": "Processing", "preview_url": f"/uploads/{safe}"})
    return {"data": resp}

@app.get("/results")
def get_results():
    return json.load(open(RESULTS_FILE, "r")) if os.path.exists(RESULTS_FILE) else []

@app.get("/summary")
def get_summary():
    if os.path.exists(RESULTS_FILE):
        data = json.load(open(RESULTS_FILE, "r"))
        return {"total": len(data), "matched": sum(1 for d in data if d["match"])}
    return {"total": 0, "matched": 0}

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 🚀 Run this if executing main.py directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
