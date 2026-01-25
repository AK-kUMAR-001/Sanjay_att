import sqlite3
import os

DB_FOLDER = "Attendance"
DB_NAME = "attendance.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

def drop_attendance_table():
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("DROP TABLE IF EXISTS attendance")
            conn.commit()
            print("Dropped 'attendance' table to remove constraints.")
        except Exception as e:
            print(f"Error dropping table: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    drop_attendance_table()
