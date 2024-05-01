import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from itertools import product

def main():
    # Load the CSV
    df = pd.read_csv('/home/vihaan/PushUpData/CombinedAnnotatedVideos/reduced_annotations.csv')
    
    # Create a column that concatenates all label bits into one string
    label_cols = ['Improper_Depth', 'Flared_Elbows', 'Sagged_Back', 'Wide_Hands']
    df['label_combination'] = df[label_cols].astype(str).agg(''.join, axis=1)

    # Generate all possible combinations of the labels
    all_combinations = [''.join(map(str, combo)) for combo in product([0, 1], repeat=4)]

    # Base directory for videos
    base_video_path = '/home/vihaan/PushUpData/CombinedAnnotatedVideos'
    
    # Create directories for each label combination within both training and validation folders
    create_directories('train', all_combinations)
    create_directories('val', all_combinations)

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)

    # Copy files to their respective directories
    distribute_files(train_df, 'train', base_video_path)
    distribute_files(val_df, 'val', base_video_path)

def create_directories(base_path, combinations):
    os.makedirs(base_path, exist_ok=True)
    for combo in combinations:
        os.makedirs(os.path.join(base_path, combo), exist_ok=True)

def distribute_files(df, base_path, base_video_path):
    for index, row in df.iterrows():
        source_path = os.path.join(base_video_path, row['Video_ID'])
        destination_path = os.path.join(base_path, row['label_combination'], row['Video_ID'])
        shutil.copy(source_path, destination_path)

if __name__ == '__main__':
    main()
