import os
import sqlite3
from cryptography.fernet import Fernet
import keyring #type:ignore
from database import set_connection, get_connection, initialize_database

SERVICE_NAME = "secure_ai_task_manager"
USERNAME = "db_encryption_key"
ENC_FILE = "data/tasks.enc"

def get_key():
    key = keyring.get_password(SERVICE_NAME, USERNAME)

    if key is None:
        key = Fernet.generate_key().decode()
        keyring.set_password(SERVICE_NAME, USERNAME, key)

    return key.encode()


def load_encrypted_db():
    if not os.path.exists(ENC_FILE):
        return None

    with open(ENC_FILE, "rb") as f:
        encrypted = f.read()

    fernet = Fernet(get_key())
    decrypted = fernet.decrypt(encrypted)

    return decrypted.decode()  


def save_encrypted_db():
    conn = get_connection()

    sql_script = "\n".join(conn.iterdump())  

    fernet = Fernet(get_key())
    encrypted = fernet.encrypt(sql_script.encode())

    os.makedirs("data", exist_ok=True)
    with open(ENC_FILE, "wb") as f:
        f.write(encrypted)


def load_or_init_database():
    sql_script = load_encrypted_db()
    conn = sqlite3.connect(":memory:",check_same_thread=False)
    set_connection(conn)

    if sql_script:
        conn.executescript(sql_script.strip())
    initialize_database()

    return conn
