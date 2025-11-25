import sqlite3
from datetime import datetime

_conn = None

def set_connection(conn):
    global _conn
    _conn = conn

def get_connection():
    return _conn

def initialize_database():
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            due_date TEXT,
            category TEXT,
            priority INTEGER DEFAULT 10,
            status TEXT DEFAULT 'todo',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()


def row_to_dict(row):
    if not row:
        return None

    keys = [
        "id",
        "title",
        "description",
        "due_date",
        "category",
        "priority",
        "status",
        "created_at",
        "updated_at"
    ]

    return dict(zip(keys, row))


def add(title, description="", status="todo", priority=10, category=None, due_date=None):
    conn = get_connection()
    curs = conn.cursor()

    now = datetime.utcnow().isoformat()

    curs.execute("""
        INSERT INTO tasks (title, description, due_date, category, priority, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, description, due_date, category, priority, status, now, now))

    conn.commit()
    return curs.lastrowid


def get_all():
    conn = get_connection()
    curs = conn.cursor()

    curs.execute("SELECT * FROM tasks ORDER BY created_at DESC")
    rows = curs.fetchall()

    return [row_to_dict(row) for row in rows]


def get_by_id(task_id):
    conn = get_connection()
    curs = conn.cursor()

    curs.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = curs.fetchone()

    return row_to_dict(row)


def update(task_id, **fields):
    if not fields:
        return False

    conn = get_connection()
    curs = conn.cursor()

    fields["updated_at"] = datetime.utcnow().isoformat()

    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [task_id]

    curs.execute(f"""
        UPDATE tasks
        SET {columns}
        WHERE id = ?
    """, values)

    conn.commit()
    return curs.rowcount > 0


def delete(task_id):
    conn = get_connection()
    curs = conn.cursor()

    curs.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()

    return curs.rowcount > 0
