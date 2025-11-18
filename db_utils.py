import sqlite3
# db_utils.py
"""
Module làm việc với SQLite:
- Khởi tạo database
- Lưu lịch sử phân loại
- Lấy danh sách lịch sử gần nhất
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

# Đường dẫn tới file DB nằm trong thư mục db/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "sentiments.db")


def get_connection():
    """
    Tạo connection tới SQLite.
    check_same_thread=False để dùng được trong Streamlit (nhiều thread).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    """
    Tạo bảng nếu chưa có.
    Bảng: sentiments(id, text, sentiment, timestamp)
    """
    os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()


def save_result(text: str, sentiment: str) -> None:
    """
    Lưu 1 bản ghi cảm xúc vào DB.
    """
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?);",
        (text, sentiment, timestamp),
    )

    conn.commit()
    conn.close()


def get_history(limit: int = 50, sentiment: str | None = None) -> List[Dict[str, Any]]:
    """
    Lấy danh sách lịch sử mới nhất.
    - limit: số bản ghi tối đa
    - sentiment: lọc theo POSITIVE / NEGATIVE / NEUTRAL, hoặc None nếu lấy tất cả
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT text, sentiment, timestamp
        FROM sentiments
    """
    params = []

    if sentiment:
        query += " WHERE sentiment = ?"
        params.append(sentiment)

    query += " ORDER BY id DESC LIMIT ?;"
    params.append(limit)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for text, sentiment, timestamp in rows:
        history.append(
            {
                "text": text,
                "sentiment": sentiment,
                "timestamp": timestamp,
            }
        )

    return history
