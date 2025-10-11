의료용 폐기물 분류 모델

이 저장소는 파인 튜닝한 convnext_tiny.fb_in22k 모델과 OCR(EasyOCR) 를 이용해 입력 이미지를 분석하는 추론 API 예제입니다.
이미지 1장을 업로드하면 아래 정보를 반환합니다.

is_bio : 바이오/비바이오 여부

coarse : 상위(대분류) 라벨

fine : 세부(소분류) 라벨

is_ocr / ocr_text : 이미지 내 텍스트 존재 여부 및 추출 텍스트

✨ 기능

하나의 엔드포인트(/predict)로 분류 + OCR 동시 수행

응답(JSON): is_bio, coarse, fine, is_ocr, ocr_text

멀티파트 업로드(multipart/form-data) 지원

📁 폴더 구조
project/
├── app.py               # 서버 엔트리 (Flask 또는 FastAPI 중 하나 사용)
├── inference.py         # 분류 + OCR 추론 로직
├── model_def.py         # MultiHead 모델 정의 (학습 시 사용한 코드로 교체)
├── requirements.txt     # 런타임 의존성 (PyTorch는 OS/CUDA별 별도 설치 권장)
├── label_maps.json      # 라벨 매핑 (학습 시 저장한 파일)
├── best.pt              # 학습된 모델 가중치 (OUT_DIR에서 복사)
└── README.md

⚡ 빠른 시작(Quickstart)

프로젝트 복사(또는 ZIP 다운로드 후 압축 해제)

다음 파일을 프로젝트 루트에 위치

best.pt (학습된 체크포인트)

label_maps.json (라벨 매핑)

필요한 경우 inference.py의 MODEL_PATH, LABEL_MAP_PATH 수정

가상환경 생성 및 의존성 설치

# macOS / Linux
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt


⚠️ PyTorch/torchvision은 OS/CUDA에 맞춰 별도 설치 권장

CPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

🚀 서버 실행
A) Flask로 실행
python app.py
# 기본: http://0.0.0.0:8080

B) FastAPI(Uvicorn)로 실행
uvicorn app:app --host 0.0.0.0 --port 8080
# 개발 모드(자동 리로드): --reload


현재 app.py에 사용한 프레임워크(Flask/FastAPI)에 맞는 명령을 선택하세요.
FastAPI를 사용하면 http://localhost:8080/docs 에서 바로 테스트 가능합니다.

🧪 API 테스트
curl -X POST -F "file=@/absolute/path/to/test.jpg" http://localhost:8080/predict

응답 예시
{
  "is_bio": true,
  "coarse": "soft_textile",
  "fine": "cotton",
  "is_ocr": true,
  "ocr_text": "Sterile Cotton Pad"
}

📝 참고 사항

모델 정의: model_def.py의 MultiHead는 학습 시 사용했던 실제 클래스 코드로 교체해야 합니다.
모델의 forward는 아래 형태의 딕셔너리를 반환해야 합니다:

{
  "bin":    Tensor[B, 2],          # (옵션) USE_BINARY_HEAD일 때
  "coarse": Tensor[B, n_coarse],
  "fine":   Tensor[B, n_fine]
}


라벨 매핑: label_maps.json의 키가 정수/문자열 어느 쪽이든 inference.py에서 자동 처리합니다.

대용량 체크포인트 관리: best.pt가 크면 Git LFS/HuggingFace Hub/S3/GCS 등 외부 스토리지에 올리고, 서버 시작 시 다운로드하도록 변경할 수 있습니다.

배포: 운영 환경에서는 FastAPI + Uvicorn/Gunicorn 조합(또는 Docker/Cloud Run) 권장.

OCR 언어: 한국어 OCR이 필요하면 inference.py의 EasyOCR 초기화 시 언어 목록을 ["en", "ko"] 등으로 조정하세요.