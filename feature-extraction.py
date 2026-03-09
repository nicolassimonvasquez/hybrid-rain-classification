import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure the script can see your 'models' and 'utils' folders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import models

# global dcp logic
def get_global_haze_score(frame):
    """Simplified version of DCP to get one score per frame"""
    if frame is None: return 0.5
    
    # Dark Channel
    min_channel = np.min(frame, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(min_channel, kernel)
    
    # Atmospheric Light (A)
    flat_dark = dark.ravel()
    top_num = max(int(flat_dark.size * 0.001), 1)
    top_indices = np.argpartition(flat_dark, -top_num)[-top_num:]
    A = np.max(frame.reshape(-1, 3)[top_indices]) 
    A = max(A, 1.0) 
    
    # Transmission Map (t)
    norm_frame = frame.astype(np.float32) / float(A)
    min_norm = np.min(norm_frame, axis=2)
    dark_norm = cv2.erode(min_norm, kernel)
    t = 1 - 0.95 * dark_norm
    
    return np.mean(t)

# HYBRID FEATURE EXTRACTOT DCP + R3D-18
def extract_dataset(data_root, checkpoint_path, output_csv="hybrid_features_v1.csv"):
    device = torch.device("cpu") 
    
    cols = [f'r3d_{i}' for i in range(512)] + ['dcp_score', 'label']

    if not os.path.exists(output_csv):
        pd.DataFrame(columns=cols).to_csv(output_csv, index=False)
        print(f"Created new CSV file: {output_csv}")
    else:
        existing_rows = len(pd.read_csv(output_csv))
        print(f"Found existing CSV with {existing_rows} rows. Appending new data...")

    # Initialize Model
    print("Initializing Recognizer3D...")
    model = models.Recognizer3D(
        backbone="r3d_18", 
        cls_head="mlh", 
        num_labels=3,
        is_3d=True, 
        dropout_rate=0.5
    )
    
    # Load weights
    print(f"Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    
    #Force Global Average Pooling to get 512-dim feature vector
    model.cls_head = nn.Sequential(
        nn.AdaptiveAvgPool3d(1), 
        nn.Flatten()
    )
    model.eval()

    class_map = {'clear': 0, 'light': 1, 'heavy': 2}

    for label_name, label_idx in class_map.items():
        folder = os.path.join(data_root, label_name)
        if not os.path.exists(folder):
            print(f"Folder missing: {folder}, skipping...")
            continue
        
        all_frames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpeg','.jpg'))])
        print(f"\nProcessing {len(all_frames)} NEW frames in '{label_name}'...")

        # Process in clips of 16 frames
        for i in tqdm(range(0, len(all_frames) - 16, 16)):
            clip_names = all_frames[i : i+16]
            clip_tensors = []
            dcp_values = []

            for f_name in clip_names:
                img_path = os.path.join(folder, f_name)
                bgr = cv2.imread(img_path)
                if bgr is None: continue
                
                # Calculate Physics Feature (DCP)
                dcp_values.append(get_global_haze_score(bgr))
                
                # Prepare for Deep Learning (R3D)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (224, 224))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                clip_tensors.append(tensor)

            if len(clip_tensors) < 16: continue

            # R3D-18 Forward Pass (Deep Features)
            input_video = torch.stack(clip_tensors).permute(1, 0, 2, 3).unsqueeze(0)
            with torch.no_grad():
                output = model(input_video) 
                deep_features = output.cpu().numpy().flatten()
                
                # Handle potential output variations to ensure exactly 512 features
                if len(deep_features) > 512:
                    deep_features = deep_features.reshape(512, -1).mean(axis=1)

            # Combine: 512 (R3D) + 1 (Avg DCP for clip) + 1 (Label) = 514 total columns
            combined_row = np.hstack([deep_features, [np.mean(dcp_values)], [label_idx]])
            
            # Appends data to existing CSV without overwriting
            row_df = pd.DataFrame([combined_row], columns=cols)
            row_df.to_csv(output_csv, mode='a', header=False, index=False)

    final_count = len(pd.read_csv(output_csv))
    print(f"\nCSV now contains {final_count} total samples.")

if __name__ == "__main__":
    # Update these paths to match your current setup
    BASE_DIR = "/home/nsvasquez/Desktop/Visibility System"
    
    # Points to where your NEW images are
    DATA_ROOT = f"{BASE_DIR}/r3d18_final_dataset/images" 
    
    # Points to your best model weights
    CHECKPOINT = f"{BASE_DIR}/checkpoints_batch4/checkpoint_epoch_4.pth"
    
    # Simple check before starting
    if not os.path.exists(DATA_ROOT) or not os.path.exists(CHECKPOINT):
        print(f"ERROR: Paths not found. DATA_ROOT: {DATA_ROOT} or CHECKPOINT: {CHECKPOINT}")
    else:
        extract_dataset(DATA_ROOT, CHECKPOINT)