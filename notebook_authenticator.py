import base64
import hashlib
import json
import pickle
import sqlite3
from typing import List, Dict

class NotebookAuthenticator:
    def __init__(self, db_path: str = "submissions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS submissions
                     (student_id TEXT, notebook_name TEXT, timestamp TEXT, 
                      fingerprint TEXT, cell_encodings TEXT)''')
        conn.commit()
        conn.close()

    def fingerprint_notebook(self, notebook_path: str) -> Dict[str, str]:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)

        # Generate overall fingerprint
        nb_content = json.dumps(nb['cells'])
        fingerprint = hashlib.sha256(nb_content.encode()).hexdigest()

        # Encode individual cells
        cell_encodings = []
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell_content = ''.join(cell['source'])
                encoding = base64.b64encode(cell_content.encode()).decode()
                cell_encodings.append(encoding)

        return {
            'fingerprint': fingerprint,
            'cell_encodings': json.dumps(cell_encodings)
        }

    def save_submission(self, student_id: str, notebook_name: str, notebook_path: str):
        fingerprint_data = self.fingerprint_notebook(notebook_path)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO submissions 
                     (student_id, notebook_name, timestamp, fingerprint, cell_encodings)
                     VALUES (?, ?, datetime('now'), ?, ?)''',
                  (student_id, notebook_name, 
                   fingerprint_data['fingerprint'], 
                   fingerprint_data['cell_encodings']))
        conn.commit()
        conn.close()

    def compare_submissions(self, student_id1: str, student_id2: str, notebook_name: str) -> float:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT fingerprint, cell_encodings FROM submissions
                     WHERE student_id = ? AND notebook_name = ?
                     ORDER BY timestamp DESC LIMIT 1''', (student_id1, notebook_name))
        submission1 = c.fetchone()

        c.execute('''SELECT fingerprint, cell_encodings FROM submissions
                     WHERE student_id = ? AND notebook_name = ?
                     ORDER BY timestamp DESC LIMIT 1''', (student_id2, notebook_name))
        submission2 = c.fetchone()

        conn.close()

        if not submission1 or not submission2:
            return 0.0

        fingerprint_similarity = 1 if submission1[0] == submission2[0] else 0
        
        cells1 = json.loads(submission1[1])
        cells2 = json.loads(submission2[1])
        
        cell_similarities = sum(c1 == c2 for c1, c2 in zip(cells1, cells2)) / max(len(cells1), len(cells2))

        return (fingerprint_similarity + cell_similarities) / 2

def authenticate_notebook(student_id: str, notebook_name: str, notebook_path: str):
    authenticator = NotebookAuthenticator()
    authenticator.save_submission(student_id, notebook_name, notebook_path)
    print(f"Submission for {student_id} saved successfully.")