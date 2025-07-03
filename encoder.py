import os
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

BP_DIR = 'datapoints/BPfilter'
META_PATH = 'datapoints/metadata.xlsx'

# Task label mapping
label_map = {
    '1': 'BEO',
    '2': 'CLH',
    '3': 'CRH',
    '4': 'DLF',
    '5': 'PLF',
    '6': 'DRF',
    '7': 'PRF',
    '8': 'Rest',
}

rows = []
srno = 1

# Encoding for task labels
label_encoding = {
    'BEO': 0,
    'CLH': 1,
    'CRH': 2,
    'DLF': 3,
    'PLF': 4,
    'DRF': 5,
    'PRF': 6,
    'Rest': 7,
}

def patient_sort_key(folder):
    match = re.match(r'f_S(\d+)', folder)
    return int(match.group(1)) if match else float('inf')

for patient_folder in sorted(os.listdir(BP_DIR), key=patient_sort_key):
    if not patient_folder.startswith('f_'):
        continue
    # Extract patient number
    match_patient = re.match(r'f_S(\d+)', patient_folder)
    if not match_patient:
        continue
    patient_num = int(match_patient.group(1))
    patient_path = os.path.join(BP_DIR, patient_folder)
    for file in sorted(os.listdir(patient_path)):
        if not file.startswith('bp_') or not file.endswith('.csv'):
            continue
        # Remove prefix for parsing
        base = file[3:-4]  # remove 'bp_' and '.csv'
        # Example: S24R1I6_5
        match = re.match(r'S(\d+)R(\d+)([MI])(\d+)_(\d+)', base)
        if not match:
            continue
        subj_id, rep_num, task_type, label_num, task_rep_num = match.groups()
        local_url = os.path.join(patient_path, file)
        # task_type: 1 for motor, 0 for imagery
        task_type_val = 1 if task_type == 'M' else 0
        label_str = label_map.get(label_num, 'Unknown')
        label_encoded = label_encoding.get(label_str, -1)
        rows.append([
            srno,
            patient_num,
            local_url,
            int(rep_num),
            task_type_val,
            int(task_rep_num),
            label_str,
            label_encoded
        ])
        srno += 1

# Create DataFrame and save
columns = [
    'srno',
    'patient_number',
    'local_url',
    'repetition_number',
    'task_type',
    'task_repetition_number',
    'task_label',
    'task_label_encoded'
]
df = pd.DataFrame(rows, columns=columns)
df.to_excel(META_PATH, index=False)
print(f"Metadata written to {META_PATH}")
