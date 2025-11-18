# 📘 Trợ lý phân loại cảm xúc tiếng Việt

Ứng dụng phân loại cảm xúc câu tiếng Việt sử dụng mô hình Transformer



# 🚀 Giới thiệu

Đây là ứng dụng phân loại cảm xúc tiếng Việt (Positive – Neutral – Negative), được xây dựng cho đồ án môn Seminar Chuyên Đề.

Ứng dụng cho phép:

Nhập câu tiếng Việt tuỳ ý

Chuẩn hoá câu (xử lý viết tắt, không dấu, từ lóng…)

Phân loại cảm xúc bằng mô hình Transformer pre-trained

Hiển thị kết quả theo màu sắc trực quan

Lưu lịch sử vào SQLite

Lọc lịch sử theo cảm xúc

Tải thêm lịch sử (pagination đơn giản)

Ứng dụng chạy giao diện bằng Streamlit, nhẹ, dễ dùng và chạy độc lập.



# 🛠 Công nghệ sử dụng

Python 3.10+

HuggingFace Transformers

Mô hình sentiment: 5CD-AI/vietnamese-sentiment-visobert

Underthesea (xử lý tiếng Việt)

Streamlit (UI)

SQLite (lịch sử phân loại)



# 📦 Cài đặt môi trường
1. Clone dự án hoặc tải zip
git clone <repo_url>

2. Tạo môi trường ảo (khuyến khích)
python -m venv venv


Kích hoạt:

Windows

venv\Scripts\activate


MacOS / Linux

source venv/bin/activate

3. Cài thư viện
pip install -r requirements.txt



# ▶️ Chạy ứng dụng

Chạy lệnh:

streamlit run app.py


Sau đó trình duyệt sẽ tự mở tại:

http://localhost:8501



# 📌 Tính năng chính
1. Nhập câu tiếng Việt

Hỗ trợ không dấu

Hỗ trợ viết tắt (ko → không, dc → được…)

Phát hiện câu vô nghĩa và cảnh báo

2. Chuẩn hoá câu

Hiển thị câu gốc và câu đã chuẩn hoá

Mapping hơn 100 từ không dấu → có dấu

3. Phân loại cảm xúc

3 nhãn: POSITIVE / NEUTRAL / NEGATIVE

Hiển thị màu:

🟢 Positive

🟡 Neutral

🔴 Negative

4. Lịch sử phân loại

Lưu vào SQLite

Hiển thị 50 bản ghi mới nhất

Nút “Tải thêm 10” để load thêm

Bộ lọc theo cảm xúc:

Positive

Neutral

Negative

Tất cả



# 🧪 Bộ test case

Ứng dụng kèm theo 10 test case chuẩn trong file test_cases.csv (theo yêu cầu đồ án).

Ví dụ:

Câu	Nhãn mong đợi
Hôm nay tôi rất vui	POSITIVE
Món ăn này dở quá	NEGATIVE
Thời tiết bình thường	NEUTRAL

Độ chính xác yêu cầu tối thiểu: ≥ 65%



# 📁 Cấu trúc thư mục
DoAn_SentimentAssistant/
│
├─ app.py                 # App Streamlit chính
├─ nlp_utils.py           # Xử lý tiếng Việt + NLP model
├─ db_utils.py            # SQLite helper
├─ requirements.txt       # Danh sách packages
├─ README.md              # File hướng dẫn này
├─ test_cases.csv         # Bộ test case
│
├─ db/
│   └─ sentiments.db      # Database lịch sử
│
├─ docs/
│   ├─ BaoCao_DoAn.pdf    # Báo cáo
│   └─ BaoCao_DoAn.docx   # Báo cáo
│
└─ demo/
    └─ video_demo.mp4     # Video trình bày



# 📝 Yêu cầu đầu ra (Chuẩn theo đề bài)

Ứng dụng trả kết quả dạng dictionary như sau:

{
    "text": "Bạn khỏe không?",
    "sentiment": "POSITIVE"
}



# 📚 Tài liệu tham khảo

HuggingFace Transformers

Mô hình ViSoBERT

Underthesea

Streamlit



# 🎉 Ghi chú

Ứng dụng không cần fine-tuning, dùng pipeline để đơn giản hóa.

Mapping tiếng Việt được tối ưu cho mục đích bài tập, không phải công cụ NLP hoàn chỉnh.

Phần giao diện, màu sắc và icon được tuỳ chỉnh để giúp báo cáo đẹp và dễ thuyết trình.
