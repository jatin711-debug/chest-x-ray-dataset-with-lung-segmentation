# import pandas as pd

# # Read the CSV files
# masks_df = pd.read_csv('./CXLSeg-mask.csv')
# metadata_df = pd.read_csv('./CXLSeg-metadata.csv')
# segmented_df = pd.read_csv('./CXLSeg-segmented.csv')
# split_df = pd.read_csv('./CXLSeg-split.csv')

# # Define the common keys to merge on
# common_keys = ['dicom_id', 'subject_id', 'study_id']

# # Merge the dataframes
# merged_df = masks_df.merge(metadata_df, on=common_keys, how='outer')
# merged_df = merged_df.merge(segmented_df, on=common_keys, how='outer')
# merged_df = merged_df.merge(split_df, on=common_keys, how='outer')

# # Remove duplicates based on the common keys
# merged_df.drop_duplicates(subset=common_keys, keep='first', inplace=True)

# # Optional: Reset the index after merging and dropping duplicates
# merged_df.reset_index(drop=True, inplace=True)

# # Save the merged dataframe to a new CSV file
# merged_df.to_csv('merged_CXLSeg_data.csv', index=False)

# # Print some information about the merged dataset
# print("Merged Dataset Information:")
# print(f"Total number of rows: {len(merged_df)}")
# print("\nColumns in the merged dataset:")
# print(merged_df.columns.tolist())

# # Optional: Display the first few rows to verify the merge
# print("\nFirst few rows of the merged dataset:")
# print(merged_df.head())