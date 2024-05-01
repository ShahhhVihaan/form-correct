import pandas as pd
import os
import shutil
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

    # Split the data and distribute files
    for combo in all_combinations:
        combo_df = df[df['label_combination'] == combo]
        if len(combo_df) >= 13:
            # Only create a validation set if there are at least 13 videos
            train_df, val_df = custom_train_val_split(combo_df, n_val=13)
            distribute_files(train_df, 'train', base_video_path)
            distribute_files(val_df, 'val', base_video_path)

def create_directories(base_path, combinations):
    os.makedirs(base_path, exist_ok=True)
    for combo in combinations:
        os.makedirs(os.path.join(base_path, combo), exist_ok=True)

def distribute_files(df, base_path, base_video_path):
    for index, row in df.iterrows():
        source_path = os.path.join(base_video_path, row['Video_ID'])  # Ensure correct file extension
        destination_path = os.path.join(base_path, row['label_combination'], row['Video_ID'] + '.mp4')
        shutil.copy(source_path, destination_path)

def custom_train_val_split(df, n_val):
    # Ensure the validation set has exactly n_val videos
    val_df = df.sample(n=n_val, random_state=42)
    train_df = df.drop(val_df.index)
    return train_df, val_df

if __name__ == '__main__':
    main()
