import os
import shutil
import pandas as pd
import re

# Define paths
dataset1_root = r"C:\Users\user\.cache\kagglehub\datasets\meowmeowmeowmeowmeow\gtsrb-german-traffic-sign\versions\1\Train"
dataset2_root = r"C:\Users\user\.cache\kagglehub\datasets\ahemateja19bec1025\traffic-sign-dataset-classification\versions\2\traffic_Data\DATA"
labels1_path = r"C:\Users\user\Documents\github\learning-tf\traffic-sign\sign_dic.csv"
labels2_path = r"C:\Users\user\.cache\kagglehub\datasets\ahemateja19bec1025\traffic-sign-dataset-classification\versions\2\labels.csv"
output_dir = r"C:\Users\user\.cache\kagglehub\datasets\merged_datasets\data"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

def extract_speed(desc):
    match = re.search(r'\d+', desc)
    return int(match.group()) if match else None

# Load and process dataset1 labels
df1 = pd.read_csv(labels1_path)
global_labels = {}  # Maps global class IDs to descriptions
description_to_global = {}  # Maps descriptions to global class IDs
max_global_id = -1

for _, row in df1.iterrows():
    class_id = str(row['id']).strip()
    desc = row['description'].strip()
    global_labels[class_id] = desc
    description_to_global[desc] = class_id
    current_id = int(class_id)
    if current_id > max_global_id:
        max_global_id = current_id

# Load and process dataset2 labels
df2 = pd.read_csv(labels2_path)
dataset2_mapping = {}  # Maps dataset2 class IDs to global class IDs
next_id = max_global_id + 1

for _, row in df2.iterrows():
    class_id = str(row['ClassId']).strip()
    desc = row['Name'].strip()
    speed = extract_speed(desc)
    global_id = None

    # Check if description exists in dataset1
    if desc in description_to_global:
        global_id = description_to_global[desc]
    else:
        # Assign a new global ID
        global_id = str(next_id)
        global_labels[global_id] = desc
        description_to_global[desc] = global_id
        next_id += 1

    dataset2_mapping[class_id] = global_id

# Copy images from dataset1
for class_folder in os.listdir(dataset1_root):
    src_dir = os.path.join(dataset1_root, class_folder)
    if not os.path.isdir(src_dir):
        continue
    dest_dir = os.path.join(output_dir, class_folder)
    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy(src, dst)

# Copy images from dataset2
for class_folder in os.listdir(dataset2_root):
    src_dir = os.path.join(dataset2_root, class_folder)
    if not os.path.isdir(src_dir):
        continue
    global_id = dataset2_mapping.get(class_folder, None)
    if global_id is None:
        continue  # Skip if not mapped
    dest_dir = os.path.join(output_dir, global_id)
    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy(src, dst)

# Generate global_labels.csv
global_df = pd.DataFrame(list(global_labels.items()), columns=['id', 'description'])
global_df = global_df.sort_values(by='id')
global_csv_path = os.path.join(output_dir, 'global_labels.csv')
global_df.to_csv(global_csv_path, index=False)

print("Datasets merged successfully. Global labels saved to:", global_csv_path)