# nlp_utils.py
"""
Module xử lý NLP:
- Chuẩn hoá câu tiếng Việt
- Gọi mô hình Transformer để phân loại cảm xúc
"""

from functools import lru_cache
import re

from underthesea import word_tokenize
from transformers import pipeline

# Mô hình sentiment tiếng Việt trên HuggingFace
MODEL_NAME = "5CD-AI/vietnamese-sentiment-visobert"

LABEL_MAPPING = {
    "NEG": "NEGATIVE",
    "POS": "POSITIVE",
    "NEU": "NEUTRAL",
}


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    """
    Chỉ load model 1 lần duy nhất (dùng cache) để tránh bị chậm.
    """
    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )
    return sentiment_pipeline

VIET_VOWELS = set(
    "aeiouyAEIOUY"
    "ăâêôơưĂÂÊÔƠƯ"
    "áàảãạÁÀẢÃẠ"
    "ắằẳẵặẮẰẲẴẶ"
    "ấầẩẫậẤẦẨẪẬ"
    "éèẻẽẹÉÈẺẼẸ"
    "óòỏõọÓÒỎÕỌ"
    "ốồổỗộỐỒỔỖỘ"
    "ớờởỡợỚỜỞỠỢ"
    "íìỉĩịÍÌỈĨỊ"
    "úùủũụÚÙỦŨỤ"
    "ýỳỷỹỵÝỲỶỸỴ"
    "đĐ"
)

VIET_STOPWORDS = {
    "là", "và", "của", "không", "rất", "này", "kia", "đó", "này",
    "tôi", "ban", "bạn", "mình", "anh", "em", "chỉ", "thì",
    "nhưng", "nếu", "vì", "nên", "cho", "khi", "đã", "đang", "sẽ",
    "ở", "trong", "trên", "với", "hay", "cũng", "rồi", "luôn",
}


def is_valid_vietnamese(text: str) -> bool:
    """
    Heuristic đơn giản:
    - Có ít nhất 2 từ
    - Phần lớn kí tự là chữ cái, có đủ nguyên âm tiếng Việt
    - Có ít nhất 1 stopword Việt phổ biến
    - Không toàn là kí tự ngẫu nhiên / số / ký hiệu
    """
    if not isinstance(text, str):
        return False

    text = text.strip()
    if len(text) < 3:
        return False

    # Lấy các "từ" là cụm chữ cái
    words = re.findall(r"[A-Za-zÀ-Ỹà-ỹ]+", text)
    if len(words) < 2:
        return False

    letters = "".join(words)
    if not letters:
        return False

    # Tỉ lệ nguyên âm
    vowel_count = sum(1 for ch in letters if ch in VIET_VOWELS)
    ratio_vowel = vowel_count / len(letters)

    # Nếu quá ít nguyên âm → thường là random / không phải tiếng Việt
    if ratio_vowel < 0.25:
        return False

    # Nếu có ít nhất 1 stopword Việt → ưu tiên coi là hợp lệ
    lower_words = [w.lower() for w in words]
    if any(w in VIET_STOPWORDS for w in lower_words):
        return True

    # Trung bình độ dài từ nếu quá dài → dễ là chuỗi vô nghĩa
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 10:
        return False

    # Mặc định: tạm coi là hợp lệ
    return True


def normalize_text(text: str) -> str:
    """
    Chuẩn hoá câu để HIỂN THỊ cho người dùng.
    Ví dụ: 'Ban khoe ko?' -> 'Bạn khỏe không?'
    (Chỉ làm 1 số luật đơn giản cho demo, không phải phục hồi dấu 100%)
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip().lower()

    # Tách token (giữ cả dấu ? ! , .)
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    # Một số mapping cơ bản
    mapping = {
        # Đại từ – Ngôi xưng
        "toi": "tôi",
        "ban": "bạn",
        "minh": "mình",
        "anh": "anh",
        "chi": "chị",
        "em": "em",
        "co": "cô",
        "chu": "chú",
        "ba": "bà",
        "ong": "ông",
        "nguoi": "người",
        "ho": "họ",

        # Động từ – tính từ phổ biến
        "yeu": "yêu",
        "thuong": "thương",
        "ghet": "ghét",
        "thich": "thích",
        "biet": "biết",
        "hieu": "hiểu",
        "thay": "thấy",
        "khoe": "khỏe",
        "om": "ốm",
        "dau": "đau",
        "met": "mệt",
        "vui": "vui",
        "buon": "buồn",
        "gian": "giận",
        "nong": "nóng",
        "lanh": "lạnh",
        "dep": "đẹp",
        "xau": "xấu",

        # Từ phủ định – viết tắt
        "khong": "không",
        "k": "không",
        "ko": "không",
        "k0": "không",
        "hok": "không",
        "khg": "không",
        "hk": "không",
        "kh": "không",
        
        # Câu hỏi
        "gi": "gì",
        "j": "gì",
        "sao": "sao",
        "tai": "tại",
        "vi": "vì",
        "tai sao": "tại sao",

        # Các trạng từ
        "rat": "rất",
        "hon": "hơn",
        "lam": "lắm",
        "qua": "quá",
        "nhieu": "nhiều",
        "it": "ít",
        "noi": "nói",
        "noi chuyen": "nói chuyện",

        # Địa điểm – thời gian
        "nay": "nay",
        "mai": "mai",
        "hom": "hôm",
        "truoc": "trước",
        "sau": "sau",
        "o": "ở",

        # Từ liên kết
        "va": "và",
        "voi": "với",
        "vi": "vì",
        "nen": "nên",
        "nhung": "nhưng",

        # Các từ cơ bản
        "duoc": "được",
        "dc": "được",
        "du": "đủ",
        "thoi": "thôi",
        "roi": "rồi",
        "cung": "cũng",
        "luon": "luôn",
        "neu": "nếu",
        "dang": "đang",
        "se": "sẽ",
        "da": "đã",

        # Danh từ
        "con": "con",
        "nguoi": "người",
        "ban be": "bạn bè",
        "gia dinh": "gia đình",
        "cong viec": "công việc",
        "truong": "trường",
        "lop": "lớp",
        "mon": "món",
        "an": "ăn",
        "quan": "quán",
        "nha": "nhà",
        "cua": "của",

        # Cảm xúc – đánh giá
        "te": "tệ",
        "tot": "tốt",
        "hay": "hay",
        "do": "dở",
        "chap nhan": "chấp nhận",
        "tuyet": "tuyệt",

        # Chat / internet slang
        "ad": "admin",
        "ib": "nhắn",
        "rep": "trả lời",
        "like": "thích",
        "sub": "đăng ký",
        "vid": "video",

        # Từ có dấu phổ biến bị gõ sai TELEX
        "thuc": "thực",
        "phai": "phải",
        "thuan": "thuận",
        "mien": "miền",
        "quoc": "quốc",
        "dong": "đông",
        "tay": "tây",
        "nam": "năm",
        "troi": "trời",
        "muon": "muốn",

        # Từ nối dài
        "cam on": "cảm ơn",
        "xin loi": "xin lỗi",
        "tam biet": "tạm biệt",
        "chao": "chào",

        # Thường gặp khi không sử dụng dấu
        "kha": "khá",
        "de": "dễ",
        "kho": "khó",
        "to": "to",
        "nho": "nhỏ",
        "lon": "lớn",
        "nhe": "nhẹ",
        "man": "mặn",
        "ngot": "ngọt",
    }


    new_tokens = []
    for tok in tokens:
        if tok.isalpha():  # chỉ map với chữ cái
            mapped = mapping.get(tok, tok)
            new_tokens.append(mapped)
        else:
            new_tokens.append(tok)

    # Ghép lại, xử lý khoảng trắng trước dấu câu
    sentence = ""
    for tok in new_tokens:
        if tok in [".", ",", ":", ";", "?", "!", "…"]:
            sentence = sentence.rstrip() + tok + " "
        else:
            sentence += tok + " "
    sentence = sentence.strip()

    # Viết hoa chữ cái đầu
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    return sentence


def preprocess(text: str) -> str:
    """
    Chuẩn hoá câu để đưa vào mô hình Transformer:
    - đưa về chữ thường
    - thay viết tắt cơ bản
    - tách từ bằng underthesea
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip().lower()
    replacements = {
        "ko": "không",
        "kô": "không",
        "khong": "không",
        "hok": "không",
        "k": "không",
        "k0": "không",
        "zui": "vui",
        "sz": "size",
        "dc": "được",
        "đc": "được",
        "vs": "với",
        "j": "gì",
        "0": "không",
        "nma": "nhưng mà",
        "nhg": "nhưng",
        "mn": "mọi người",
        "hk": "không",
        "thik": "thích",
        "hoy": "không",
        "hoi": "không",
    }
    for k, v in replacements.items():
        text = text.replace(f" {k} ", f" {v} ")

    tokens = word_tokenize(text, format="text")
    return tokens


def classify(text: str) -> dict:
    """
    Phân loại cảm xúc cho 1 câu tiếng Việt.
    Trả về dict:
    {
        "original_text": ...,
        "normalized_text": ...,
        "sentiment": ...,
        "score": ...
    }
    """
    if not text or not text.strip():
        raise ValueError("Câu nhập vào rỗng.")
    if not is_valid_vietnamese(text):
        raise ValueError("Câu nhập vào không giống tiếng Việt hoặc không có nghĩa rõ ràng.")


    # Câu chuẩn hoá để hiển thị cho người dùng
    normalized = normalize_text(text)

    # Câu chuẩn hoá cho mô hình
    cleaned = preprocess(text)
    sentiment_pipeline = get_sentiment_pipeline()

    result = sentiment_pipeline(cleaned)[0]
    raw_label = result.get("label", "").upper()

    sentiment = LABEL_MAPPING.get(raw_label, "NEUTRAL")
    score = float(result.get("score", 0.0))

    return {
        "original_text": text,
        "normalized_text": normalized,
        "sentiment": sentiment,
        "score": score,
    }
