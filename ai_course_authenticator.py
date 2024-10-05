import base64
import hashlib
import json
import csv
import zipfile
import os
from typing import Dict, List, Any

class AICourseAuthenticator:
    def __init__(self, output_dir: str = "submissions"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def encode_notebook(self, notebook_path: str) -> Dict[str, Any]:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)

        encoded_cells = []
        implemented_methods = []
        estimations = {}

        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell_content = ''.join(cell['source'])
                encoded_cell = base64.b64encode(cell_content.encode()).decode()
                encoded_cells.append(encoded_cell)

                # Extract implemented methods (assuming they start with 'def')
                methods = [line.strip() for line in cell_content.split('\n') if line.strip().startswith('def ')]
                implemented_methods.extend(methods)

                # Extract estimations (assuming they're commented with '# Estimation:')
                for line in cell_content.split('\n'):
                    if '# Estimation:' in line:
                        key, value = line.split('# Estimation:')[-1].split(':')
                        estimations[key.strip()] = float(value.strip())

        return {
            'encoded_cells': encoded_cells,
            'implemented_methods': implemented_methods,
            'estimations': estimations
        }

    def create_submission_csv(self, data: Dict[str, Any], csv_path: str):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Type', 'Value'])
            
            for method in data['implemented_methods']:
                writer.writerow(['Method', method])
            
            for key, value in data['estimations'].items():
                writer.writerow(['Estimation', f"{key}: {value}"])

    def create_submission_zip(self, student_id: str, notebook_path: str):
        encoded_data = self.encode_notebook(notebook_path)
        
        submission_dir = os.path.join(self.output_dir, student_id)
        os.makedirs(submission_dir, exist_ok=True)
        
        # Save encoded notebook
        encoded_notebook_path = os.path.join(submission_dir, 'encoded_notebook.json')
        with open(encoded_notebook_path, 'w') as f:
            json.dump(encoded_data, f)
        
        # Create submission.csv
        csv_path = os.path.join(submission_dir, 'submission.csv')
        self.create_submission_csv(encoded_data, csv_path)
        
        # Create zip file
        zip_path = os.path.join(self.output_dir, f"{student_id}_submission.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(notebook_path, os.path.basename(notebook_path))
            zipf.write(encoded_notebook_path, os.path.basename(encoded_notebook_path))
            zipf.write(csv_path, os.path.basename(csv_path))
        
        print(f"Submission for {student_id} saved successfully as {zip_path}")

    def compare_submissions(self, student_id1: str, student_id2: str) -> float:
        def load_encoded_data(student_id):
            path = os.path.join(self.output_dir, student_id, 'encoded_notebook.json')
            with open(path, 'r') as f:
                return json.load(f)

        data1 = load_encoded_data(student_id1)
        data2 = load_encoded_data(student_id2)

        # Compare encoded cells
        cell_similarity = sum(c1 == c2 for c1, c2 in zip(data1['encoded_cells'], data2['encoded_cells'])) / max(len(data1['encoded_cells']), len(data2['encoded_cells']))

        # Compare implemented methods
        method_similarity = len(set(data1['implemented_methods']) & set(data2['implemented_methods'])) / max(len(data1['implemented_methods']), len(data2['implemented_methods']))

        # Compare estimations
        estimation_keys = set(data1['estimations'].keys()) & set(data2['estimations'].keys())
        estimation_similarity = sum(abs(data1['estimations'][k] - data2['estimations'][k]) < 1e-6 for k in estimation_keys) / max(len(data1['estimations']), len(data2['estimations']))

        return (cell_similarity + method_similarity + estimation_similarity) / 3

def authenticate_notebook(student_id: str, notebook_path: str="./main.ipynb"):
    authenticator = AICourseAuthenticator()
    authenticator.create_submission_zip(student_id, notebook_path)