import os
data_path = "/home/nsvasquez/Desktop/Visibility System/visibility-system/.venv/python/r3d18_final_dataset/images"
folders = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
for i, name in enumerate(folders):
    print(f"Index {i} is mapped to folder: {name}")