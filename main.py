from encrypted_storage import load_or_init_database, save_encrypted_db
from database import get_connection
import atexit

def initialize():
    load_or_init_database()

def save():
    save_encrypted_db()

def main():
    initialize()
    atexit.register(save)
    print("Database initialized in memory.")
    print("Run the Streamlit UI with:  streamlit run app.py")

if __name__ == "__main__":
    main()
