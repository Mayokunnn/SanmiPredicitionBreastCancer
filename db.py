import sqlite3

def create_db():
    conn = sqlite3.connect('specimens.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS specimens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            image_size REAL,
            magnification REAL,
            real_size REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(username, image_size, magnification, real_size):
    conn = sqlite3.connect('specimens.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO specimens (username, image_size, magnification, real_size)
        VALUES (?, ?, ?, ?)
    ''', (username, image_size, magnification, real_size))
    conn.commit()
    conn.close()

create_db()
