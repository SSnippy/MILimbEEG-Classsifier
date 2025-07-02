import os
import pandas as pd
import re

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
        is_motor = 1 if task_type == 'M' else 0
        is_imagery = 1 if task_type == 'I' else 0
        label_str = label_map.get(label_num, 'Unknown')
        rows.append([
            srno,
            patient_num,
            local_url,
            int(rep_num),
            is_motor,
            is_imagery,
            int(task_rep_num),
            label_str
        ])
        srno += 1

# Create DataFrame and save
columns = [
    'srno',
    'patient_number',
    'local_url',
    'repetition_number',
    'is_motor_task',
    'is_motor_imagery_task',
    'task_repetition_number',
    'task_label'
]
df = pd.DataFrame(rows, columns=columns)
df.to_excel(META_PATH, index=False)
print(f"Metadata written to {META_PATH}")
