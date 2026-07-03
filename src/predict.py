from __future__ import annotations

import argparse
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.utils import load_config


def predict_email(raw_text: str, model_dir: str = "models/distilbert") -> dict:
    """Predict phishing using the fine-tuned DistilBERT model.

    Args:
        raw_text: The original email text.
        model_dir: Path to the saved DistilBERT directory.

    Returns:
        dict with keys: 'label' (int), 'label_name' (str), 'confidence' (float).
    """
    from pathlib import Path
    if not Path(model_dir).exists() and model_dir == "models/distilbert":
        print(f"⚠️ Không tìm thấy thư mục '{model_dir}'. Tự động tải mô hình gốc 'distilbert-base-uncased' từ HuggingFace...")
        model_dir = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cfg = load_config()
    max_length = cfg.get("transformer", {}).get("max_length", 512)

    encoding = tokenizer(
        raw_text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()

    return {
        "label": pred,
        "label_name": "phishing" if pred == 1 else "safe",
        "confidence": confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict phishing from email text using DistilBERT.")
    parser.add_argument("text", type=str, help="Email text to classify")
    parser.add_argument("--model", default="models/distilbert", help="Path to DistilBERT model directory")
    args = parser.parse_args()

    result = predict_email(args.text, args.model)
    print(f"Prediction: {result['label_name']} (confidence: {result['confidence']:.2%})")


if __name__ == "__main__":
    main()

