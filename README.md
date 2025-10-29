**의료용 폐기물 분류 모델**

의료용 폐기물 이미지를 분류하고 텍스트를 인식하는 FastAPI 기반 추론 서버입니다.
이 저장소는 파인 튜닝한 convnext_tiny.fb_in22k 모델과 OCR(EasyOCR) 를 이용해 입력 이미지를 분석하는 추론 API 예제입니다.
이미지 1장을 업로드하면 아래 정보를 반환합니다.

## 🔧 기술 스택

- **FastAPI**: API 서버 프레임워크
- **PyTorch**: 딥러닝 모델 추론
- **ConvNeXt**: 이미지 분류 모델 백본
- **EasyOCR**: 텍스트 인식
- **Git LFS**: 대용량 모델 파일 관리

## 📥 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/Labify-DAMO/ai.git
cd ai
```

2. **Git LFS 파일 다운로드**
```bash
git lfs install
git lfs pull
```

3. **가상환경 설정**
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

4. **의존성 설치**
```bash
pip install -r requirements.txt
# GPU(CUDA 12.1) 있는 경우:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU 전용인 경우:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 🚀 서버 실행

```bash
# 개발 모드
uvicorn app:app --host 0.0.0.0 --port 8080 --reload

# 프로덕션 모드
uvicorn app:app --host 0.0.0.0 --port 8080
```

## 📚 API 명세

### 1. 상태 확인
- **엔드포인트**: `GET /health`
- **응답**: 
```json
{
    "status": "ok"
}
```

### 2. 이미지 분류 및 OCR
- **엔드포인트**: `POST /predict`
- **요청**: `multipart/form-data`
  - `file`: 이미지 파일 (jpg, png, jpeg)
- **응답**:
```json
{
    "is_bio": true,          // 바이오/비바이오 여부
    "coarse": "주사기류",    // 대분류
    "fine": "주사기",        // 소분류
    "is_ocr": true,          // 텍스트 존재 여부
    "ocr_text": "검출된 텍스트"
}
```

## 📁 프로젝트 구조
```
project/
├── app.py               # FastAPI 서버 엔트리포인트
├── inference.py         # 모델 로딩 및 추론 로직
├── model_def.py         # 모델 아키텍처 정의
├── requirements.txt     # 패키지 의존성
├── best.pt             # 학습된 모델 가중치 (Git LFS)
├── label_maps.json     # 레이블 매핑 정보 (Git LFS)
└── README.md
```

## ⚙️ 환경 설정

### 모델 설정
- `MODEL_PATH`: 모델 가중치 파일 경로 (`inference.py`)
- `LABEL_MAP_PATH`: 레이블 매핑 파일 경로 (`inference.py`)

### OCR 설정
- 기본 지원 언어: 영어, 한국어
- 언어 변경: `inference.py`의 `Reader(["en", "ko"])` 부분 수정

## 🔍 테스트 방법

1. **cURL 사용**
```bash
curl -X POST "http://localhost:8080/predict" -F "file=@테스트이미지.jpg"
```

2. **Python 요청**
```python
import requests

url = "http://localhost:8080/predict"
files = {"file": open("테스트이미지.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## ⚠️ 주의사항

1. GPU 메모리 요구사항: 최소 4GB 이상 권장
2. 지원 이미지 형식: jpg, png, jpeg
3. 최대 이미지 크기: 10MB


## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
