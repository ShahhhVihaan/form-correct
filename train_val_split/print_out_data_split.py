import os

def count_videos(directory):
    video_counts = []
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for class_name in classes:
        class_path = os.path.join(directory, class_name)
        # Count only .mp4 files
        count = sum(1 for f in os.listdir(class_path) if f.endswith('.mp4'))
        video_counts.append((class_name, count))
    return video_counts

def print_video_counts(train_counts, val_counts):
    # Map class names to counts
    train_dict = dict(train_counts)
    val_dict = dict(val_counts)
    
    # Get a unique list of all class names
    all_classes = set(train_dict.keys()).union(set(val_dict.keys()))
    
    # Print the counts
    for class_name in sorted(all_classes):
        train_num = train_dict.get(class_name, 0)
        val_num = val_dict.get(class_name, 0)
        print(f'{class_name} {train_num} {val_num}')

# Path to the root directory containing train and val folders
root_path = '/home/vihaan/Projects/form-correct/final_datasets/second_iter'

# Count videos in the train and val directories
train_videos = count_videos(os.path.join(root_path, 'train'))
val_videos = count_videos(os.path.join(root_path, 'val'))

# Print results
print_video_counts(train_videos, val_videos)
