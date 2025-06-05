import pandas as pd

# Load your CSVs
df_segmented = pd.DataFrame()
df_metadata = pd.DataFrame()
df_masks = pd.DataFrame()
df_split = pd.DataFrame()
# Ensure you have the correct filenames and paths for your CSV files
try:
    df_segmented = pd.read_csv('./CXLSeg-segmented.csv')
    df_metadata = pd.read_csv('./CXLSeg-metadata.csv') # Or your actual metadata filename
    df_masks = pd.read_csv('./CXLSeg-mask.csv')
    df_split = pd.read_csv('./CXLSeg-split.csv') # Or your actual split filename
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}. Please ensure filenames are correct.")
    # Handle error appropriately

# Rename DicomPath columns for clarity before merging if they are both named 'DicomPath'
# It's good practice if they point to different things (image vs. mask)
df_segmented = df_segmented.rename(columns={'DicomPath': 'ImagePath'})
df_masks = df_masks.rename(columns={'DicomPath': 'MaskPath'})

# Merge dataframes
# Start with the file that has your primary labels and image paths (df_segmented)
merged_df = pd.merge(df_segmented, df_metadata, on=['dicom_id', 'subject_id', 'study_id'], how='left')
merged_df = pd.merge(merged_df, df_masks, on=['dicom_id', 'subject_id', 'study_id'], how='left')
merged_df = pd.merge(merged_df, df_split, on=['dicom_id', 'subject_id', 'study_id'], how='left')


#save the merged DataFrame to a new CSV file
merged_df.to_csv('./CXLSeg-merged.csv', index=False)


# Display the first few rows and columns to verify
print("Merged DataFrame head:")
print(merged_df.head())
print("\nMerged DataFrame columns:")
print(merged_df.columns)

# Identify pathology label columns
pathology_columns = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

