import os
import pandas as pd
from scipy.signal import butter, filtfilt

# Parameters (adjust as needed)
RAW_DIR = 'datapoints/raw'
BP_DIR = 'datapoints/BPfilter'
LOWCUT = 1.0
HIGHCUT = 50.0
FS = 125  # Sampling frequency in Hz

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

if not os.path.exists(BP_DIR):
    os.makedirs(BP_DIR)


for patient in os.listdir(RAW_DIR):
    patient_path = os.path.join(RAW_DIR, patient)
    if os.path.isdir(patient_path):
        new_patient_folder = os.path.join(BP_DIR, f"f_{patient}")
        os.makedirs(new_patient_folder, exist_ok=True)
        for file in os.listdir(patient_path):
            if file.endswith('.csv'):
                file_path = os.path.join(patient_path, file)
                print(f"Processing: {file_path}")  # Output current file being processed
                df = pd.read_csv(file_path)
                # Apply filter to all columns except non-numeric (if any)
                numeric_df = df.select_dtypes(include=['number'])
                filtered = bandpass_filter(numeric_df.values, LOWCUT, HIGHCUT, FS)
                filtered_df = pd.DataFrame(filtered, columns=numeric_df.columns)
                # If there were non-numeric columns, add them back
                for col in df.columns:
                    if col not in numeric_df.columns:
                        filtered_df[col] = df[col]
                # Save with prefix
                new_file = os.path.join(new_patient_folder, f"bp_{file}")
                filtered_df.to_csv(new_file, index=False)


print("done")