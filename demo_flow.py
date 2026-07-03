"""Demo script visualizing the exact step-by-step data flow through DistilBERT architecture.
From raw text -> Word/Position Embeddings -> Attention -> FFN/GELU -> Classification Head.

Usage:
    python demo_flow.py
"""

import os
import sys
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 80)
    print("🚀 BẮT ĐẦU PHÂN TÍCH CHI TIẾT TỪNG BƯỚC LUỒNG DỮ LIỆU CỦA DISTILBERT")
    print("=" * 80)

    # ------------------------------------------------------------------
    # BƯỚC 0: TẢI MÔ HÌNH & TỪ ĐIỂN
    # ------------------------------------------------------------------
    print("\n[BƯỚC 0] Đang tải mô hình DistilBERT từ thư mục 'models/distilbert'...")
    model_dir = "models/distilbert"
    if not os.path.exists(model_dir):
        print(f"⚠️ Không tìm thấy '{model_dir}', tự động sử dụng 'distilbert-base-uncased' từ HuggingFace Hub...")
        model_dir = "distilbert-base-uncased"

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # Chuyển mô hình sang chế độ đánh giá (evaluation mode)
    print("=> Hoàn tất tải mô hình & từ điển WordPiece.")

    # ------------------------------------------------------------------
    # BƯỚC 1: DỮ LIỆU ĐẦU VÀO
    # ------------------------------------------------------------------
    raw_email = "URGENT! Your account will be locked. Click http://spam-link.com to verify."
    print("\n" + "=" * 80)
    print("[BƯỚC 1] DỮ LIỆU ĐẦU VÀO (Raw Input Email)")
    print("=" * 80)
    print(f"💌 Văn bản gốc: '{raw_email}'")

    # ------------------------------------------------------------------
    # BƯỚC 2: TOKENIZATION (Cắt từ WordPiece)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BƯỚC 2] TOKENIZATION (Cắt từ theo từ điển WordPiece)")
    print("=" * 80)
    print("-> Bộ tách từ chia văn bản thành các từ/căn tố (sub-words) và tự động thêm token đặc biệt:")
    print("   + [CLS] (ID 101): Token khởi đầu, vector của nó sẽ dùng để phân loại toàn bộ câu.")
    print("   + [SEP] (ID 102): Token đánh dấu kết thúc câu.")
    tokens = tokenizer.tokenize(raw_email)
    tokens_with_special = ["[CLS]"] + tokens + ["[SEP]"]
    print(f"👉 Danh sách Tokens ({len(tokens_with_special)} tokens): {tokens_with_special}")

    # ------------------------------------------------------------------
    # BƯỚC 3: ENCODING & PADDING (Chuyển thành ID số & Mặt nạ chú ý)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BƯỚC 3] ENCODING & PADDING (Chuyển thành Ma trận số Tensors)")
    print("=" * 80)
    encoding = tokenizer(
        raw_email,
        truncation=True,
        padding="max_length",
        max_length=32,  # Sử dụng max_length=32 để hiển thị demo cho gọn đẹp
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    print(f"👉 Ma trận input_ids (Shape: {input_ids.shape}):")
    print(f"   {input_ids.tolist()[0]}")
    print(f"👉 Ma trận attention_mask (Shape: {attention_mask.shape} - 1 là từ thật, 0 là padding [PAD]):")
    print(f"   {attention_mask.tolist()[0]}")

    # ------------------------------------------------------------------
    # BƯỚC 4: WORD EMBEDDINGS & POSITION EMBEDDINGS (Nhúng từ vựng & Vị trí)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BƯỚC 4] EMBEDDINGS LAYER (Nhúng từ vựng & Nhúng vị trí)")
    print("=" * 80)
    with torch.no_grad():
        # 4.1. Word Embeddings
        word_embeddings = model.distilbert.embeddings.word_embeddings(input_ids)
        print("-> 1. Word Embeddings: Chuyển mỗi ID số thành vector 768 chiều mang ngữ nghĩa từ vựng.")
        print(f"   + Kích thước Word Embeddings: {word_embeddings.shape} (Batch x Seq_Len x Dim_768)")

        # 4.2. Position Embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.distilbert.embeddings.position_embeddings(position_ids)
        print("-> 2. Position Embeddings: Chuyển vị trí [0, 1, ..., 31] thành vector 768 chiều.")
        print("   + Giải thích: Vì Transformer xử lý song song tất cả các từ cùng lúc, Position Embeddings giúp mô hình nhận biết thứ tự từ trong câu (từ đứng đầu vs đứng cuối).")
        print(f"   + Kích thước Position Embeddings: {position_embeddings.shape}")

        # 4.3. Layer Normalization & Dropout
        embeddings_sum = word_embeddings + position_embeddings
        embeddings_norm = model.distilbert.embeddings.LayerNorm(embeddings_sum)
        embeddings_output = model.distilbert.embeddings.dropout(embeddings_norm)
        print("-> 3. Tổng hợp Embeddings: Output = LayerNorm(Word_Embed + Position_Embed) + Dropout.")
        print(f"👉 Kích thước vector đầu ra lớp Embeddings: {embeddings_output.shape}")
        print(f"   + Giá trị mẫu vector [CLS] sau Embeddings (5 chiều đầu tiên): {embeddings_output[0, 0, :5].tolist()}")

    # ------------------------------------------------------------------
    # BƯỚC 5: LỚP TRANSFORMER 0 - MULTI-HEAD SELF-ATTENTION (Tự chú ý Đa đầu)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BƯỚC 5] TRANSFORMER LAYER 0: MULTI-HEAD SELF-ATTENTION (Tự chú ý)")
    print("=" * 80)
    with torch.no_grad():
        layer_0 = model.distilbert.transformer.layer[0]
        
        # Tạo mask đúng định dạng cho attention (chuyển 0 -> -inf, 1 -> 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        print("-> Lớp Self-Attention (Tự chú ý 12 đầu - 12 Heads):")
        print("   + Tính toán ma trận Query (Q), Key (K), Value (V) từ vector đầu vào.")
        print("   + Tính Attention Scores = Softmax(Q * K^T / sqrt(d_k)): Đo lường mức độ liên quan giữa các từ.")
        print("   + Ví dụ: Từ 'account' và 'locked' sẽ có điểm chú ý (Attention score) với nhau rất cao!")
        
        # Trình diễn tính toán ma trận Query (Q), Key (K), Value (V)
        q = layer_0.attention.q_lin(embeddings_output)
        k = layer_0.attention.k_lin(embeddings_output)
        v = layer_0.attention.v_lin(embeddings_output)
        print(f"   + Ma trận Query (Q): Shape {q.shape} (Hỏi: từ này cần tìm thông tin gì?)")
        print(f"   + Ma trận Key (K):   Shape {k.shape} (Chìa khóa: từ này chứa thông tin gì?)")
        print(f"   + Ma trận Value (V): Shape {v.shape} (Giá trị: ngữ nghĩa thực sự của từ)")
        
        # Chạy qua module Attention của Layer 0
        attn_output = layer_0.attention(embeddings_output, attention_mask=extended_attention_mask)[0]
        print(f"👉 Kích thước sau Self-Attention: {attn_output.shape} (Mỗi từ đã được bổ sung ngữ cảnh từ toàn câu)")

        # Residual Connection 1 & LayerNorm 1
        sa_output = layer_0.sa_layer_norm(embeddings_output + attn_output)
        print("-> Kết nối dư (Residual Connection) & LayerNorm Lần 1: X_norm1 = LayerNorm(X + Attention(X))")

    # ------------------------------------------------------------------
    # BƯỚC 6: LỚP TRANSFORMER 0 - FEED-FORWARD NETWORK (FFN) & GELU ACTIVATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BƯỚC 6] TRANSFORMER LAYER 0: FEED-FORWARD NETWORK (FFN) & HÀM GELU")
    print("=" * 80)
    with torch.no_grad():
        print("-> Mạng truyền thẳng (FFN) gồm 2 lớp Linear kết hợp hàm kích hoạt phi tuyến GELU:")
        
        # 6.1. Linear 1 (Expand 768 -> 3072)
        ffn_lin1_out = layer_0.ffn.lin1(sa_output)
        print(f"   1. Lớp Linear 1: Mở rộng chiều vector từ 768 lên 3,072 chiều. Kích thước: {ffn_lin1_out.shape}")
        
        # 6.2. GELU Activation
        ffn_gelu_out = layer_0.ffn.activation(ffn_lin1_out)
        print("   2. Hàm kích hoạt GELU (Gaussian Error Linear Unit):")
        print("      + Khác với ReLU (cắt cứng dưới 0), GELU làm mượt các giá trị âm theo phân phối xác suất Gaussian, giúp mô hình học các đặc trưng phức tạp và mượt mà hơn.")
        print(f"      + Giá trị trước GELU (3 số đầu): {ffn_lin1_out[0, 0, :3].tolist()}")
        print(f"      + Giá trị sau GELU  (3 số đầu): {ffn_gelu_out[0, 0, :3].tolist()}")

        # 6.3. Linear 2 (Project back 3072 -> 768)
        ffn_lin2_out = layer_0.ffn.lin2(ffn_gelu_out)
        print(f"   3. Lớp Linear 2: Nén vector từ 3,072 về lại 768 chiều. Kích thước: {ffn_lin2_out.shape}")

        # Residual Connection 2 & LayerNorm 2
        layer_0_output = layer_0.output_layer_norm(sa_output + ffn_lin2_out)
        print("-> Kết nối dư (Residual Connection) & LayerNorm Lần 2: Output_Layer0 = LayerNorm(X_norm1 + FFN(X_norm1))")
        print(f"👉 Hoàn tất 1 lớp Transformer! Kích thước đầu ra: {layer_0_output.shape}")

    # ------------------------------------------------------------------
    # BƯỚC 7: LẶP QUA 6 LỚP TRANSFORMER & TRÍCH XUẤT [CLS] TOKEN
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BƯỚC 7] LẶP QUA 6 LỚP TRANSFORMER & TRÍCH XUẤT VECTOR [CLS]")
    print("=" * 80)
    with torch.no_grad():
        # Đưa qua toàn bộ 6 lớp Transformer (Layer 0 đến Layer 5)
        distilbert_output = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = distilbert_output.last_hidden_state
        print(f"-> Dữ liệu đi tuần tự qua 6 lớp Transformer (Layer 0 -> Layer 5) để tinh luyện ngữ nghĩa.")
        print(f"👉 Kích thước Last Hidden State (Đầu ra lớp thứ 6): {last_hidden_state.shape} (Batch x Seq x 768)")

        # Trích xuất vector của token [CLS] (vị trí index 0)
        cls_vector = last_hidden_state[:, 0]
        print(f"\n-> Trích xuất vector [CLS] tại vị trí index 0 đại diện cho toàn bộ câu email:")
        print(f"👉 Kích thước vector [CLS]: {cls_vector.shape} (Nén toàn bộ email thành 1 vector 768 chiều)")
        print(f"   + Giá trị mẫu vector [CLS] (5 chiều đầu): {cls_vector[0, :5].tolist()}")

    # ------------------------------------------------------------------
    # BƯỚC 8: CLASSIFICATION HEAD (Pre-classifier -> ReLU -> Classifier -> Logits)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BƯỚC 8] CLASSIFICATION HEAD (Lớp Phân loại cuối cùng)")
    print("=" * 80)
    with torch.no_grad():
        # 8.1. Pre-classifier Linear(768, 768) + ReLU + Dropout
        pre_classifier_out = model.pre_classifier(cls_vector)
        pre_classifier_relu = torch.relu(pre_classifier_out)
        print("-> 1. Pre-classifier: Linear(768 -> 768) + Hàm kích hoạt ReLU + Dropout.")
        print(f"   + Kích thước sau Pre-classifier: {pre_classifier_relu.shape}")

        # 8.2. Classifier Linear(768, 2)
        logits = model.classifier(pre_classifier_relu)
        print("-> 2. Classifier (Lớp tuyến tính cuối): Linear(768 -> 2 chiều tương ứng 2 nhãn).")
        print(f"👉 Kích thước Logits output: {logits.shape}")
        print(f"   + Giá trị Logits thô: [Safe = {logits[0, 0].item():.4f}, Phishing = {logits[0, 1].item():.4f}]")

    # ------------------------------------------------------------------
    # BƯỚC 9: TÍNH XÁC SUẤT SOFTMAX & KẾT LUẬN
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[BƯỚC 9] TÍNH XÁC SUẤT BẰNG HÀM SOFTMAX & KẾT LUẬN")
    print("=" * 80)
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        safe_prob = probs[0, 0].item()
        phish_prob = probs[0, 1].item()
        
        print("-> Lớp Softmax chuẩn hóa Logits thành xác suất phần trăm (tổng bằng 100%):")
        print(f"   🟢 Xác suất Safe Email (An toàn):   {safe_prob * 100:.2f}%")
        print(f"   🔴 Xác suất Phishing Email (Lừa đảo): {phish_prob * 100:.2f}%")

        pred_idx = torch.argmax(probs, dim=-1).item()
        pred_label = "PHISHING EMAIL (Email Lừa Đảo)" if pred_idx == 1 else "SAFE EMAIL (Email An Toàn)"
        conf = phish_prob if pred_idx == 1 else safe_prob

        print("\n" + "★" * 80)
        print(f"🎯 KẾT QUẢ DỰ ĐOÁN CUỐI CÙNG: ** {pred_label} **")
        print(f"📊 Độ tự tin của mô hình (Confidence Score): {conf * 100:.2f}%")
        print("★" * 80)

    print("\n🎉 HOÀN TẤT PHÂN TÍCH LUỒNG DỮ LIỆU DISTILBERT!")
    print("=" * 80)


if __name__ == "__main__":
    main()
