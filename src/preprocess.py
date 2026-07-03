from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Preprocess email text for exploratory data analysis (EDA) and classical ML models.

    Design choices for phishing detection:
    - Lowercase for normalization.
    - Replace URLs with <URL> token (preserves signal that URLs exist without
      letting specific URLs confuse the Transformer embeddings).
    - Normalize whitespace.
    - Keep punctuation (exclamation marks, question marks are phishing signals).
    """
    text = text.lower()
    # Replace URLs with a token so the Transformer captures "has URL" as a feature
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
