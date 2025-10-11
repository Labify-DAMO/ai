import cv2, torch, easyocr, json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_def import MultiHead
import timm

# --- Paths (override if needed) ---
MODEL_PATH = "best.pt"
LABEL_MAP_PATH = "label_maps.json"

# --- Preprocessing (match your validation transforms) ---
val_tfms = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(224, 224, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

def _label_from_map(idx_map, idx):
    """Safely get label by handling int or str keys."""
    # If keys are strings, use str(idx); if ints, use idx
    try:
        return idx_map[idx]
    except KeyError:
        return idx_map[str(idx)]

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_maps = json.load(f)
    n_coarse = len(label_maps["coarse_to_idx"])
    n_fine   = len(label_maps["fine_to_idx"])

    # ✅ 학습 때와 동일하게 backbone 인스턴스를 만들어서 전달
    backbone = timm.create_model(
        'convnext_tiny.fb_in22k',
        pretrained=False,             # checkpoint에서 가중치 로드할 거라 False
        num_classes=0,
        global_pool='avg'
    )

    model = MultiHead(
        backbone=backbone,
        use_binary=True,
        n_coarse=n_coarse,
        n_fine=n_fine,
        use_supervised_contam=False,
        n_contam=0
    )

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)

    reader = easyocr.Reader(["en","ko"], gpu=torch.cuda.is_available())
    return model, label_maps, reader, device

def predict_image(path, model, label_maps, reader, device):
    # Read image (BGR -> RGB)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Classification
    t = val_tfms(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu")):
        out = model(t)

        # Binary head
        is_bio = True
        if "bin" in out:
            probs_bin = torch.softmax(out["bin"], dim=1)
            is_bio = probs_bin[0, 1].item() > 0.5
            if not is_bio:
                return {"is_bio": False, "is_ocr": False, "ocr_text": ""}

        # Coarse / Fine
        coarse_idx = int(torch.argmax(out["coarse"], dim=1).item())
        fine_idx   = int(torch.argmax(out["fine"],   dim=1).item())

        coarse_label = _label_from_map(label_maps["idx_to_coarse"], coarse_idx)
        fine_label   = _label_from_map(label_maps["idx_to_fine"],   fine_idx)

    # OCR
    # Using the RGB image directly; EasyOCR expects RGB in ndarray
    ocr_texts = reader.readtext(img, detail=0)  # only texts
    ocr_text = " ".join([t for t in ocr_texts if isinstance(t, str)]).strip()
    is_ocr = bool(ocr_text)

    return {
        "is_bio": True,
        "coarse": coarse_label,
        "fine": fine_label,
        "is_ocr": is_ocr,
        "ocr_text": ocr_text
    }
