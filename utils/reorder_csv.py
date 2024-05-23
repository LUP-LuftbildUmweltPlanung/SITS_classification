"""
This script reorders CSV files based on the 'aoi' column from the 'meta.csv' file.
It filters files based on a specified keyword (e.g., "augsburg" or "2022") and distributes
them into 'test' and 'train' folders within a newly created directory structure.

Instructions:
1. Set the `keyword` variable to the city name or year you are interested in.
2. Make sure the directories in the `source_folder` variable exist and you have read/write permissions.
3. Run the script. It will copy the CSV files into the new 'test' and 'train' directories without
   altering the original files, preserving the original data structure.

Dependencies:
- pandas for data manipulation.
- shutil and os for file and directory operations.
- tqdm for displaying the progress bar.
"""

import pandas as pd
import shutil
import os
from tqdm import tqdm

def count_files_in_directory(directory):
    """Utility function to count files in a given directory"""
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# Define paths
meta_csv_path = '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/envilink_vv_3years_09/meta.csv'
source_folder = '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/envilink_vv_3years_09/sepfiles'
new_root_folder = source_folder.replace("envilink_vv_3years_09", "envilink_vv_3years_vechta")
# User specifies the keyword (e.g., "augsburg" or "2022")
keyword = 'vechta'  # Change this to your needed filter

test_folder_source = os.path.join(source_folder, 'test/csv')
train_folder_source = os.path.join(source_folder, 'train/csv')
test_folder = os.path.join(new_root_folder, 'test/csv')
train_folder = os.path.join(new_root_folder, 'train/csv')

# Ensure the new directories exist
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)

# Load the meta CSV file
meta_df = pd.read_csv(meta_csv_path)

# Count files in original folders before copying
original_test_count_source = count_files_in_directory(test_folder_source)
original_train_count_source = count_files_in_directory(train_folder_source)
print(f"Original Source: {original_test_count_source} files in 'test', {original_train_count_source} files in 'train'")

# Filter function to determine test set based on 'aoi' column
def filter_test_set(row, keyword):
    return keyword.lower() in row['aoi'].lower()

# Apply filter to determine the test set
meta_df['is_test'] = meta_df.apply(lambda row: filter_test_set(row, keyword), axis=1)

# Iterate with progress bar
for _, row in tqdm(meta_df.iterrows(), total=meta_df.shape[0], desc="Processing files"):
    original_file = f"{row['global_idx']}.csv"
    original_path_test = os.path.join(test_folder_source, original_file)
    original_path_train = os.path.join(train_folder_source, original_file)

    # Determine the source path
    if os.path.exists(original_path_test):
        original_path = original_path_test
    elif os.path.exists(original_path_train):
        original_path = original_path_train
    else:
        tqdm.write(f"File {original_file} not found in test or train folders.")
        continue  # Skip this file if not found

    if row['is_test']:
        # Copy to new test folder
        destination_path = os.path.join(test_folder, original_file)
    else:
        # Copy to new train folder
        destination_path = os.path.join(train_folder, original_file)

    shutil.copy(original_path, destination_path)
    #tqdm.write(f"Copied {original_file} to {'test' if row['is_test'] else 'train'} folder")

# Count files in new folders
new_test_count = count_files_in_directory(test_folder)
new_train_count = count_files_in_directory(train_folder)
print(f"New: {new_test_count} files in 'test', {new_train_count} files in 'train'")

print("Files have been reordered and placed in the new directory structure based on the AOI criteria.")
