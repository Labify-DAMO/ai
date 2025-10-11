from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import os, tempfile

from inference import load_model, predict_image

# 1) lifespan에서 모델 1회 로딩
_model_bundle = None  # (model, label_maps, reader, device)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_bundle
    _model_bundle = load_model()  # 서버 시작 시 1회 로딩
    try:
        yield
    finally:
        # 종료 시 정리할 리소스 있으면 여기서 정리
        pass

# 2) 앱 생성 시 lifespan 전달
app = FastAPI(title="ICT Classifier + OCR", version="1.0", lifespan=lifespan)

# 3) CORS (필요 시 도메인으로 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    is_bio: bool
    coarse: Optional[str] = None
    fine: Optional[str] = None
    is_ocr: bool
    ocr_text: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # 업로드 파일을 임시 저장 후 추론
    suffix = os.path.splitext(file.filename or "")[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_image(tmp_path, *_model_bundle)
        return PredictResponse(
            is_bio=bool(result.get("is_bio", False)),
            coarse=result.get("coarse"),
            fine=result.get("fine"),
            is_ocr=bool(result.get("is_ocr", False)),
            ocr_text=result.get("ocr_text", ""),
        )
    finally:
        try: os.remove(tmp_path)
        except: pass

