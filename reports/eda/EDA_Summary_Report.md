# BÁO CÁO TỔNG HỢP PHÂN TÍCH THĂM DÒ DỮ LIỆU (EDA)

## 1. Khảo sát và làm sạch dữ liệu (Data Cleaning)
- **Tổng số dòng dữ liệu thô (Raw CSV):** `18,650` dòng.
- **Số dòng bị rỗng/lỗi (Null/NaN):** `16` dòng -> Đã tiến hành loại bỏ để không gây lỗi quá trình Tokenization.
- **Số dòng trùng lặp nội dung (Duplicates):** `1,097` dòng.
- **Số lượng mẫu hợp lệ sử dụng chính thức:** `18,634` mẫu email.

## 2. Phân phối nhãn dữ liệu (Class Distribution)
| Nhãn dữ liệu | Số lượng mẫu | Tỷ lệ phần trăm |
| :--- | :---: | :---: |
| **Safe Email (0 - An toàn)** | 11,322 | 60.76% |
| **Phishing Email (1 - Lừa đảo)** | 7,312 | 39.24% |
| **Tổng cộng** | **18,634** | **100.00%** |

## 3. Thống kê độ dài văn bản & Ngưỡng cắt gọt (Truncation 512 Tokens)
- **Độ dài trung bình (Số từ):** Safe Email = `~686` từ | Phishing Email = `~302` từ.
- **Độ dài trung bình (Số tokens):** Safe Email = `~762` tokens | Phishing Email = `~348` tokens.
- **Tỷ lệ vượt ngưỡng 512 tokens:** Có `3,644` mẫu (19.56%) có độ dài lớn hơn 512 tokens.
- **Phương pháp xử lý:** Sử dụng cơ chế `truncation=True, max_length=512` của `DistilBertTokenizerFast`. Vì phần lớn thông tin đe dọa, lời kêu gọi hành động (Call to action) và đường dẫn lừa đảo thường xuất hiện ở phần đầu/giữa email, việc cắt gọt phần đuôi dài không làm ảnh hưởng đến độ chính xác (Vẫn đạt F1 97.20%).

## 4. Phân tích đặc trưng thao túng tâm lý & Cú pháp (Urgency & Punctuation)
- **Tỷ lệ chứa từ khóa khẩn cấp/đe dọa (`urgent`, `verify`, `account`, `suspend`, `bank`, `security`...):** Safe Email (`23.16%`) vs Phishing Email (`43.53%`). Phishing Email sử dụng từ khóa thúc giục cao gấp đôi so với email thông thường.
- **Tỷ lệ sử dụng dấu chấm than (`!`):** Safe Email (`21.75%`) vs Phishing Email (`58.07%`). Có tới gần 60% email lừa đảo chèn dấu chấm than để gây chú ý.
- **Số lượng dấu chấm than (`!`) trung bình:** Phishing Email sử dụng trung bình `3.3` dấu `!` mỗi thư, gấp nhiều lần so với Safe Email (`1.9`), phản ánh rõ nét nỗ lực tạo áp lực tâm lý khẩn cấp (Urgency/Fear).
- **Lưu ý về mã HTML:** Khi thực hiện khảo sát EDA, nhóm phát hiện các thẻ HTML thô (`<html>`, `<body>`, `<a href>`) trong bộ dữ liệu Kaggle đã được tiền xử lý loại bỏ từ trước (tỷ lệ xấp xỉ 0%), do đó nhóm tập trung khai thác các đặc trưng ngữ nghĩa khẩn cấp và dấu câu mang lại tín hiệu phân loại cao nhất cho kiến trúc Transformer.
