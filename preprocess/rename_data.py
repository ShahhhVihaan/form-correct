import os
import subprocess

def test_and_rename_files(directory, prefix):
    files = [f for f in os.listdir(directory) if f.endswith('.mp4')]    
    count = 1

    for file in files:
        print(file)
        full_path = os.path.join(directory, file)
        try:
            subprocess.run(['open', full_path], check=True, timeout=10)  # Timeout in seconds
        except subprocess.CalledProcessError:
            print(f"Failed to open {file}, leaving unchanged.")
            continue
        except subprocess.TimeoutExpired:
            print(f"Timeout expired for {file}, leaving unchanged.")
            continue
        
        # Construct new file name with a sequential counter and prefix
        new_name = f"{prefix}{count}.mp4"
        new_full_path = os.path.join(directory, new_name)
        
        os.rename(full_path, new_full_path)
        print(f"Renamed {file} to {new_name}")
        
        count += 1

# Usage example: test_and_rename_files('/path/to/your/directory', 'YourPrefix')

test_and_rename_files("/home/vihaan/PushUpData/KaggleDataset/Wrong", "C")