import pandas as pd
import json
import os

# --- Configuration ---
# !!! IMPORTANT: Please verify these paths and column names !!!
CSV_FILE_NAME = 'CXLSeg-merged.csv'  # Change if your main CSV has a different name
OUTPUT_JSONL_FILE_NAME = 'cxlseg_finetuning_dataset.jsonl'
WORKSPACE_ROOT = os.getcwd() # Assumes script is run from the workspace root

# Define the abnormality columns based on your provided list
ABNORMALITY_COLUMNS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices"
]

# Define essential columns expected in the CSV
IMAGE_PATH_COL = 'ImagePath'
REPORTS_COL = 'Reports'
DICOM_PATH = 'MaskPath'
SUBJECT_ID_COL = 'subject_id' # Optional, for inclusion
STUDY_ID_COL = 'study_id'     # Optional, for inclusion
SPLIT_COL = 'split'           # Optional, for inclusion
# --- End Configuration ---

def create_finetuning_dataset(max_entries=None):
    """
    Reads the CXLSeg CSV data, processes it, and writes it to a JSONL file
    suitable for fine-tuning multimodal models.
    If max_entries is provided, only that many entries will be written.
    """
    csv_file_path = os.path.join(WORKSPACE_ROOT, CSV_FILE_NAME)
    output_jsonl_path = os.path.join(WORKSPACE_ROOT, OUTPUT_JSONL_FILE_NAME)

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please ensure CSV_FILE_NAME in the script is correct and the file exists.")
        return

    print(f"Reading CSV file from: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Verify necessary columns exist
    required_input_columns = [IMAGE_PATH_COL, REPORTS_COL]
    missing_cols = [col for col in required_input_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: The CSV file is missing the following essential columns: {', '.join(missing_cols)}")
        return

    # Check for abnormality columns (warn if any are missing, but proceed)
    for col in ABNORMALITY_COLUMNS:
        if col not in df.columns:
            print(f"Warning: Abnormality column '{col}' not found in the CSV. It will be skipped.")

    processed_count = 0
    print(f"Processing data and writing to: {output_jsonl_path}")
    with open(output_jsonl_path, 'w') as outfile:
        for index, row in df.iterrows():
            if max_entries is not None and processed_count >= max_entries:
                break
            try:
                image_path = row[IMAGE_PATH_COL]
                report = row[REPORTS_COL]
                dicom_path = row.get(DICOM_PATH, None)  # Optional DICOM path

                # Extract abnormalities
                abnormalities_present = []
                for col in ABNORMALITY_COLUMNS:
                    if col in df.columns and pd.notna(row[col]) and row[col] == 1:
                        abnormalities_present.append(col)

                # Handle cases where 'Reports' might be missing (NaN)
                if pd.isna(report):
                    report = ""  # Use empty string for missing reports

                # Create the JSON object for the current row
                data_entry = {
                    "image_path": image_path,
                    "report": report,
                    "dicom_path": dicom_path if dicom_path else None,
                    "abnormalities": abnormalities_present
                }

                # Add optional fields if they exist in the CSV
                if SUBJECT_ID_COL in df.columns and pd.notna(row[SUBJECT_ID_COL]):
                    data_entry["subject_id"] = row[SUBJECT_ID_COL]
                if STUDY_ID_COL in df.columns and pd.notna(row[STUDY_ID_COL]):
                    data_entry["study_id"] = row[STUDY_ID_COL]
                if SPLIT_COL in df.columns and pd.notna(row[SPLIT_COL]):
                    data_entry["split"] = row[SPLIT_COL]
                
                outfile.write(json.dumps(data_entry) + '\n')
                processed_count += 1
            except Exception as e:
                print(f"Error processing row {index}: {e}. Skipping this row.")
                continue
                
    print(f"Dataset successfully created at {output_jsonl_path}")
    print(f"Total entries processed and written: {processed_count} out of {len(df)} original rows.")

if __name__ == '__main__':
    # Example: generate only 100 entries
    create_finetuning_dataset(max_entries=100)
