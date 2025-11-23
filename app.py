import sqlite3
# app.py
"""
Ứng dụng Streamlit:
- Nhận câu tiếng Việt từ người dùng
- Gọi NLP để phân loại cảm xúc
- Lưu lịch sử vào SQLite
- Hiển thị lịch sử phân loại
"""

import streamlit as st

from db_utils import init_db, save_result, get_history
from nlp_utils import classify

# Khởi tạo DB ngay khi run app
init_db()
# State cho lịch sử
if "history_limit" not in st.session_state:
    st.session_state["history_limit"] = 50  # mặc định 50 bản ghi

if "history_filter" not in st.session_state:
    st.session_state["history_filter"] = "ALL"  # ALL / POSITIVE / NEGATIVE / NEUTRAL


st.set_page_config(
    page_title="Trợ lý phân loại cảm xúc tiếng Việt",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Trợ lý phân loại cảm xúc tiếng Việt")
st.write(
    "Nhập một câu tiếng Việt bất kỳ. Ứng dụng sẽ phân loại cảm xúc thành "
    "**POSITIVE**, **NEUTRAL** hoặc **NEGATIVE**."
)

st.markdown("---")

# Nhập liệu
user_text = st.text_area(
    "Nhập câu tiếng Việt:",
    height=120,
    placeholder="Ví dụ: Hôm nay tôi rất vui vì được 10 điểm...",
)

col1, col2 = st.columns([1, 1])

with col1:
    classify_btn = st.button("Phân loại cảm xúc")


# Kết quả phân loại
if classify_btn:
    if not user_text or len(user_text.strip()) == 0:
        st.error("❗ Câu nhập vào đang trống. Vui lòng nhập nội dung.")
    elif len(user_text.strip()) < 5:
        st.warning("⚠ Câu hơi ngắn, vui lòng nhập câu rõ nghĩa hơn (>= 5 ký tự).")
    else:
        with st.spinner("Đang phân tích cảm xúc..."):
            try:
                result = classify(user_text)

                original_text = result["original_text"]
                normalized_text = result["normalized_text"]
                sentiment = result["sentiment"]
                score = result["score"]

                # Lưu vào DB (lưu câu gốc)
                save_result(original_text, sentiment)

                # ===== Hiển thị câu gốc & câu chuẩn hoá =====
                st.write("**Câu gốc:** ", original_text)
                st.write("**Câu chuẩn hoá:** ", normalized_text)

                # ===== Hiển thị cảm xúc theo màu + icon =====
                color_map = {
                    "POSITIVE": ("🟢", "TÍCH CỰC", "✓", "green"),
                    "NEGATIVE": ("🔴", "TIÊU CỰC", "✗", "red"),
                    "NEUTRAL": ("🟡", "TRUNG TÍNH", "❓", "gold"),
                }

                icon, label_vi, symbol, color = color_map.get(
                    sentiment, ("⚪", "KHÔNG RÕ", "?", "gray")
                )

                st.markdown(
                    f"""
                    <div style='padding:12px;border-radius:8px;border:1px solid {color};background-color:#fdfdfd; margin-top:8px;'>
                        <h3 style='color:{color};margin:0;'>{icon} {label_vi} ({symbol})</h3>
                        <p style='margin:4px 0;'>Độ tin cậy: <b>{score:.2f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # (tuỳ chọn) Hiển thị đúng kiểu "dictionary 2 trường" như đề
                st.subheader("Đầu ra dạng dictionary:")
                st.json({
                    "text": normalized_text,
                    "sentiment": sentiment
                })

            except ValueError as e:
                # Lỗi do mình chủ động raise (câu vô nghĩa / không phải tiếng Việt)
                st.error(f"❗ {e}")
            except Exception as e:
                # Lỗi kỹ thuật khác
                st.error(f"Đã xảy ra lỗi kỹ thuật khi phân loại: {e}")



st.markdown("---")
st.subheader("📜 Lịch sử phân loại")

# --- Bộ lọc + nút tải thêm ---
col_filter, col_info, col_more = st.columns([2, 1, 1])

with col_filter:
    filter_label = st.selectbox(
        "Lọc theo nhãn:",
        options=["Tất cả", "Positive", "Neutral", "Negative"],
        index=0,
    )

filter_map = {
    "Tất cả": "ALL",
    "Positive": "POSITIVE",
    "Neutral": "NEUTRAL",
    "Negative": "NEGATIVE",
}
st.session_state["history_filter"] = filter_map[filter_label]


# Xác định sentiment filter thật gửi xuống DB
sentiment_filter = (
    None if st.session_state["history_filter"] == "ALL"
    else st.session_state["history_filter"]
)

# Lấy lịch sử từ DB
history = get_history(
    limit=st.session_state["history_limit"],
    sentiment=sentiment_filter,
)

def load_more():
    st.session_state["history_limit"] += st.session_state.get("history_increment", 10)




if not history:
    st.info("Chưa có lịch sử nào khớp với bộ lọc hiện tại.")
else:
    color_map = {
        "POSITIVE": ("🟢", "green", "✓"),
        "NEGATIVE": ("🔴", "red", "✗"),
        "NEUTRAL": ("🟡", "gold", "❓"),
    }

    for item in history:
        text = item["text"]
        sentiment = item["sentiment"]
        timestamp = item["timestamp"]

        icon, color, symbol = color_map.get(sentiment, ("⚪", "gray", "?"))

        st.markdown(
            f"""
            <div style="border:1px solid {color}; padding:10px; border-radius:8px; margin-bottom:8px; background:#fdfdfd;">
                <span style="font-size:18px;">{icon}</span>
                <b style="color:{color};"> {sentiment} ({symbol})</b><br>
                <span style="font-size:14px;">📝 {text}</span><br>
                <span style="font-size:12px; color:#666;">⏱ {timestamp}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if len(history) >= st.session_state["history_limit"]:
        if "history_increment" not in st.session_state:
            st.session_state["history_increment"] = 10  
        if st.button("Tải thêm", on_click=load_more):
            st.session_state["history_limit"] += st.session_state["history_increment"]
