import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import models

# AUGMENTATION LOGIC 
def apply_sequence_augmentation(frames):
    """
    Applies consistent artifacts to a 16-frame sequence.
    The whole clip gets the same shift to preserve temporal motion.
    """
    brightness = random.uniform(0.7, 1.3)
    noise_sigma = random.uniform(0, 15)
    
    aug_frames = []
    for frame in frames:
        # Brightness artifact
        aug_img = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
        # Noise artifact
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, aug_img.shape).astype('uint8')
            aug_img = cv2.add(aug_img, noise)
        aug_frames.append(aug_img)
    return aug_frames

#PHYSICS FEATURE (DCP)
def get_global_haze_score(frame):
    """Simplified version of DCP to get one score per frame"""
    if frame is None: return 0.5
    
    min_channel = np.min(frame, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(min_channel, kernel)
    
    flat_dark = dark.ravel()
    top_num = max(int(flat_dark.size * 0.001), 1)
    top_indices = np.argpartition(flat_dark, -top_num)[-top_num:]
    A = np.max(frame.reshape(-1, 3)[top_indices]) 
    A = max(A, 1.0) 
    
    norm_frame = frame.astype(np.float32) / float(A)
    min_norm = np.min(norm_frame, axis=2)
    dark_norm = cv2.erode(min_norm, kernel)
    t = 1 - 0.95 * dark_norm
    
    return np.mean(t)

# MAIN EXTRACTION WITH RESUME CAPABILITY
def extract_dataset(data_root, checkpoint_path, output_csv="hybrid_features_90k_augmented.csv"):
    cols = [f'r3d_{i}' for i in range(512)] + ['dcp_score', 'label']

    # Initialize or Load CSV
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=cols).to_csv(output_csv, index=False)
        print(f"Created new CSV: {output_csv}")
        existing_rows_count = 0
    else:
        # We read just the label column to save RAM while checking progress
        temp_df = pd.read_csv(output_csv, usecols=['label'])
        existing_rows_count = len(temp_df)
        print(f"Resuming! {existing_rows_count} rows already exist in {output_csv}")

    # Initialize Model
    print("Initializing Recognizer3D...")
    model = models.Recognizer3D(
        backbone="r3d_18", 
        cls_head="mlh", 
        num_labels=3,
        is_3d=True, 
        dropout_rate=0.5
    )
    
    print(f"Loading weights: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    
    # Remove final head to get 512-dim features
    model.cls_head = nn.Sequential(
        nn.AdaptiveAvgPool3d(1), 
        nn.Flatten()
    )
    model.eval()

    class_map = {'clear': 0, 'light': 1, 'heavy': 2}

    for label_name, label_idx in class_map.items():
        folder = os.path.join(data_root, label_name)
        if not os.path.exists(folder):
            print(f"Skipping {label_name}, folder not found.")
            continue
        
        all_frames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpeg','.jpg'))])
        
        # Calculate Checkpoint: How many rows exist for THIS class?
        if existing_rows_count > 0:
            done_for_this_class = len(temp_df[temp_df['label'] == label_idx])
            start_idx = done_for_this_class * 4  # Stride is 4
        else:
            start_idx = 0

        if start_idx >= len(all_frames) - 16:
            print(f"{label_name} is already complete.")
            continue

        print(f"\nProcessing {label_name} from frame index {start_idx}...")

        # Process with STRIDE 4
        for i in tqdm(range(start_idx, len(all_frames) - 16, 4)):
            clip_names = all_frames[i : i+16]
            raw_frames = []
            
            for f_name in clip_names:
                img_path = os.path.join(folder, f_name)
                bgr = cv2.imread(img_path)
                if bgr is not None:
                    raw_frames.append(bgr)

            if len(raw_frames) < 16: continue

            # Apply augmentation to 30% of clips
            if random.random() < 0.3:
                proc_frames = apply_sequence_augmentation(raw_frames)
            else:
                proc_frames = raw_frames

            clip_tensors = []
            dcp_values = []

            for bgr in proc_frames:
                dcp_values.append(get_global_haze_score(bgr))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (224, 224))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                clip_tensors.append(tensor)

            # R3D Feature Extraction
            input_video = torch.stack(clip_tensors).permute(1, 0, 2, 3).unsqueeze(0)
            with torch.no_grad():
                output = model(input_video) 
                deep_features = output.cpu().numpy().flatten()
                if len(deep_features) > 512:
                    deep_features = deep_features.reshape(512, -1).mean(axis=1)

            # Save immediately to disk (Safe for crashes)
            combined_row = np.hstack([deep_features, [np.mean(dcp_values)], [label_idx]])
            pd.DataFrame([combined_row]).to_csv(output_csv, mode='a', header=False, index=False)
            
            # Optional: Small sleep to prevent CPU thermal throttling
            # time.sleep(0.01)

    print(f"\nDone! Final CSV: {output_csv}")

if __name__ == "__main__":
    # Ensure these paths match folder structure exactly
    BASE_DIR = "/home/nsvasquez/Desktop/Visibility System/hybrid-visibility-system"
    DATA_ROOT = f"{BASE_DIR}/r3d18_final_dataset/images" 
    CHECKPOINT = f"{BASE_DIR}/r3d18_checkpoints_batch4/checkpoint_epoch_4.pth"
    
    extract_dataset(DATA_ROOT, CHECKPOINT)