import os
import shutil

source_folder1 = '/Users/vihaan/Library/CloudStorage/OneDrive-UniversityofConnecticut/CV Group 5 Project/4_20 Rep Videos'
source_folder2 = '/Users/vihaan/Library/CloudStorage/OneDrive-UniversityofConnecticut/CV Group 5 Project/Kaggle Rep Videos'
destination_folder = '/Users/vihaan/Library/CloudStorage/OneDrive-UniversityofConnecticut/CV Group 5 Project/CombinedAnnotatedVideos'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Function to rename and move files
def rename_and_move_files(source_folder, prefix, destination_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith('.mp4'):
            new_name = f"{prefix}_{filename}"
            shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, new_name))

# Rename and move files from both source folders to the destination
rename_and_move_files(source_folder1, 'set1', destination_folder)
rename_and_move_files(source_folder2, 'set2', destination_folder)