import sqlite3
import pandas as pd

DB_PATH = 'data/spam_messages.db'

def initialize_db():
    """데이터베이스 초기화 및 테이블 생성"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            label INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_message(message, label):
    """새로운 메시지 저장"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO messages (message, label) VALUES (?, ?)', (message, label))
    conn.commit()
    conn.close()

def load_messages():
    """데이터베이스에서 모든 메시지 불러오기"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT message, label FROM messages', conn)
    conn.close()
    return df

