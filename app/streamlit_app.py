from __future__ import annotations

import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_config

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️")
st.title("🛡️ Phishing Email Detector")
st.write("Paste an email body below to get a prediction.")

cfg = load_config()

# ---------------------------------------------------------------
# Asset loaders (cached)
# ---------------------------------------------------------------
@st.cache_resource
def load_bert_assets(model_dir: str = "models/distilbert"):
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
    from pathlib import Path

    if not Path(model_dir).exists() and model_dir == "models/distilbert":
        model_dir = "distilbert-base-uncased"

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

# ---------------------------------------------------------------
# Email input
# ---------------------------------------------------------------
default_email = "URGENT! Your account will be locked. Click http://spam-link.com to verify."
text = st.text_area("Email text (Nhập nội dung email cần kiểm tra):", value=default_email, height=150)

if st.button("Predict & Analyze Flow (Dự đoán & Phân tích luồng)"):
    if not text.strip():
        st.warning("Please paste some email text first.")
    else:
        # ---------- DistilBERT inference & internal states ----------
        import torch

        tokenizer, model, device = load_bert_assets()
        max_length = cfg.get("transformer", {}).get("max_length", 512)

        # Step 1: Tokenize
        tokens = tokenizer.tokenize(text)
        tokens_with_special = ["[CLS]"] + tokens + ["[SEP]"]
        
        # Step 2: Encode
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=min(max_length, 32), # use 32 for clean visual presentation
            return_tensors="pt",
        )
        encoding_device = {k: v.to(device) for k, v in encoding.items()}
        input_ids = encoding_device["input_ids"]
        attention_mask = encoding_device["attention_mask"]

        with torch.no_grad():
            # Step 3: Embeddings Layer
            word_embeddings = model.distilbert.embeddings.word_embeddings(input_ids)
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
            position_embeddings = model.distilbert.embeddings.position_embeddings(position_ids)
            
            embeddings_sum = word_embeddings + position_embeddings
            embeddings_norm = model.distilbert.embeddings.LayerNorm(embeddings_sum)
            embeddings_output = model.distilbert.embeddings.dropout(embeddings_norm)

            # Step 4: Layer 0 Attention & FFN
            layer_0 = model.distilbert.transformer.layer[0]
            ext_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            ext_mask = (1.0 - ext_mask) * -10000.0
            
            attn_output = layer_0.attention(embeddings_output, attention_mask=ext_mask)[0]
            sa_output = layer_0.sa_layer_norm(embeddings_output + attn_output)
            
            ffn_lin1_out = layer_0.ffn.lin1(sa_output)
            ffn_gelu_out = layer_0.ffn.activation(ffn_lin1_out)
            ffn_lin2_out = layer_0.ffn.lin2(ffn_gelu_out)
            layer_0_output = layer_0.output_layer_norm(sa_output + ffn_lin2_out)

            # Step 5: Full 6 Layers & [CLS]
            base_output = model.distilbert(**encoding_device)
            hidden_states = base_output.last_hidden_state
            cls_vector = hidden_states[:, 0]
            
            # Step 6: Classification Head
            pre_classifier_out = model.pre_classifier(cls_vector)
            pre_classifier_relu = torch.relu(pre_classifier_out)
            logits = model.classifier(pre_classifier_relu)
            
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()

        # ---------- Display Final Prediction ----------
        st.subheader("🎯 Kết Quả Dự Đoán Cuối Cùng")
        if pred == 1:
            st.error(f"⚠️ **PHISHING EMAIL (Email Lừa Đảo)** — Độ tự tin: **{confidence:.2%}**")
        else:
            st.success(f"✅ **SAFE EMAIL (Email An Toàn)** — Độ tự tin: **{confidence:.2%}**")

        # ---------- Display Step-by-Step Data Flow ----------
        st.markdown("---")
        st.subheader("🔍 Phân Tích Chi Tiết Từng Bước Trong Kiến Trúc DistilBERT")
        st.write("Dưới đây là mô phỏng nội soi chuyên sâu từng tầng kiến trúc của mô hình DistilBERT khi xử lý email trên:")

        with st.expander("📌 Bước 1: Tokenization (Bộ tách từ WordPiece & Token đặc biệt)", expanded=True):
            st.write("Văn bản gốc được tách thành các từ hoặc căn tố (sub-words), đồng thời tự động chèn 2 token đặc biệt của BERT:")
            st.markdown("- **`[CLS]` (ID 101):** Token khởi đầu, vectơ của nó sẽ được dùng để gom thông tin phân loại cho toàn bộ email.\n- **`[SEP]` (ID 102):** Token đánh dấu kết thúc câu.")
            st.code(str(tokens_with_special), language="python")
            st.caption(f"Tổng số tokens: {len(tokens_with_special)} tokens.")

        with st.expander("📌 Bước 2: Encoding & Padding (Chuyển đổi thành ID số & Attention Mask)", expanded=False):
            st.write("Các token được ánh xạ sang ID số và làm đầy (padding) hoặc cắt gọt (truncation):")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Input IDs (Ma trận ID từ vựng):**")
                st.code(str(input_ids.tolist()[0][:16]) + " ...", language="python")
            with col2:
                st.write("**Attention Mask (Mặt nạ chú ý - 1 là từ thật, 0 là Padding):**")
                st.code(str(attention_mask.tolist()[0][:16]) + " ...", language="python")
            st.caption(f"Kích thước ma trận Tensor: `{list(input_ids.shape)}` (Batch Size x Sequence Length)")

        with st.expander("📌 Bước 3: Embeddings Layer (Word Embeddings + Position Embeddings + LayerNorm)", expanded=False):
            st.write("Đây là bước chuyển đổi ma trận ID thành vectơ không gian liên tục 768 chiều mang ngữ nghĩa:")
            st.markdown("- **Word Embeddings:** Chuyển mỗi ID từ vựng thành vectơ 768 chiều mang ngữ nghĩa từ trong từ điển.\n- **Position Embeddings:** Vì Transformer xử lý song song tất cả các từ cùng lúc, Position Embeddings bổ sung thông tin vị trí thứ tự từ (từ đứng đầu vs từ đứng cuối).\n- **LayerNorm & Dropout:** Chuẩn hóa và làm mượt vectơ tổng hợp.")
            st.code(f"Embeddings Output Shape: {list(embeddings_output.shape)}  # (Batch x Seq_Len x 768 Dimensions)", language="python")
            st.write("**Vectơ `[CLS]` mẫu sau tầng Embeddings (5 chiều đầu tiên):**")
            st.code(str([round(v, 4) for v in embeddings_output[0, 0, :5].tolist()]), language="python")

        with st.expander("📌 Bước 4: Transformer Layer 0 — Self-Attention (Tự chú ý Đa đầu)", expanded=False):
            st.write("Tại tầng Self-Attention (12 Heads), mô hình tính toán ma trận **Query (Q), Key (K), Value (V)** để tìm ra mối liên hệ và mức độ phụ thuộc ngữ cảnh giữa các từ trong email:")
            st.markdown("- **Công thức Attention:** $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$\n- **Ý nghĩa:** Ví dụ từ *'account'* sẽ tập trung chú ý rất mạnh vào từ *'locked'* và *'verify'* để nhận diện ra ngữ cảnh đe dọa.")
            st.code(f"Attention Output Shape: {list(attn_output.shape)}  # (Batch x Seq_Len x 768)", language="python")
            st.write("Sau đó đi qua kết nối dư (Residual Connection) và LayerNorm lần 1: $X_{\text{norm1}} = \text{LayerNorm}(X + \text{Attention}(X))$")

        with st.expander("📌 Bước 5: Transformer Layer 0 — Feed-Forward Network (FFN) & Hàm GELU", expanded=False):
            st.write("Sau khi chú ý ngữ cảnh, mỗi vectơ từ được đưa qua mạng truyền thẳng (FFN) gồm 2 lớp Linear và hàm kích hoạt phi tuyến **GELU**:")
            st.markdown("- **Linear 1:** Mở rộng vectơ từ 768 chiều lên **3,072 chiều** để học biểu diễn đặc trưng mức cao.\n- **Hàm kích hoạt GELU (Gaussian Error Linear Unit):** Khác với ReLU (cắt cứng dưới 0), GELU làm mượt các giá trị theo phân phối Gaussian, giúp mô hình học các đặc trưng phức tạp, tinh tế hơn.\n- **Linear 2:** Nén vectơ từ 3,072 về lại 768 chiều và áp dụng LayerNorm lần 2.")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Giá trị trước GELU (3 số đầu):**")
                st.code(str([round(v, 4) for v in ffn_lin1_out[0, 0, :3].tolist()]), language="python")
            with col_b:
                st.write("**Giá trị sau GELU (3 số đầu):**")
                st.code(str([round(v, 4) for v in ffn_gelu_out[0, 0, :3].tolist()]), language="python")
            st.caption(f"Kích thước đầu ra Transformer Layer 0: `{list(layer_0_output.shape)}`")

        with st.expander("📌 Bước 6: Lặp qua 6 Lớp Transformer & Trích xuất vectơ [CLS]", expanded=False):
            st.write("Văn bản tiếp tục đi tuần tự qua 5 lớp Transformer tiếp theo (Layer 1 đến Layer 5) để hoàn thiện việc hiểu ngữ nghĩa toàn câu:")
            st.code(f"Last Hidden State (Layer 5 Output) Shape: {list(hidden_states.shape)}", language="python")
            st.markdown("- **Trích xuất `[CLS]` Vector:** Mô hình lấy vectơ tại vị trí index 0 (tức token `[CLS]`), vì vectơ này đã tổng hợp toàn bộ thông tin ngữ cảnh của cả email thông qua cơ chế Attention.")
            st.code(f"[CLS] Vector Shape: {list(cls_vector.shape)}  # 1 vectơ 768 chiều duy nhất đại diện cho email", language="python")

        with st.expander("📌 Bước 7: Classification Head (Pre-classifier -> ReLU -> Linear -> Logits -> Softmax)", expanded=True):
            st.write("Vectơ `[CLS]` được đưa qua tầng phân loại (Classification Head) để tính xác suất cho 2 nhãn Safe và Phishing:")
            st.markdown("- **1. Pre-classifier:** Lớp Linear(768 -> 768) + Hàm kích hoạt **ReLU** + Dropout để tinh chỉnh đặc trưng trước khi phân loại.\n- **2. Classifier:** Lớp Linear(768 -> 2) để hạ về 2 chiều số học gọi là **Logits**.\n- **3. Softmax:** Chuyển đổi Logits thô thành xác suất phần trăm (%), tổng 2 nhãn bằng 100%.")
            col_s, col_p = st.columns(2)
            with col_s:
                st.metric(label="🟢 Xác suất Safe Email (An toàn)", value=f"{probs[0][0].item():.4%}")
            with col_p:
                st.metric(label="🔴 Xác suất Phishing Email (Lừa đảo)", value=f"{probs[0][1].item():.4%}")
            st.write(f"👉 **Giá trị Logits thô:** `[Safe: {logits[0][0].item():.4f}, Phishing: {logits[0][1].item():.4f}]`")


