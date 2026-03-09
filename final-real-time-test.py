"""
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Ensure models folder is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models

# --- 1. SETTINGS & PATHS ---
VIDEO_PATH = "AIC_rain.mp4" 
CHECKPOINT_R3D = "r3d18_checkpoints_batch4/checkpoint_epoch_4.pth"

# UPDATE: Point this to your new V4 directory and Epoch 47 (your best)
CHECKPOINT_MLP = "mlp_checkpoints_rigorous_v4/rigorous_epoch_13.pth" 

CLASS_NAMES = ['Clear', 'Light Rain', 'Heavy Rain']
WINDOW_SIZE = 16 
STRIDE = 4  # Matches your training data extraction stride

# --- 2. LOAD MODELS ---
device = torch.device("cpu")
print(f"Loading models on {device}...")

try:
    # Backbone: R3D18
    backbone = models.Recognizer3D(
        backbone="r3d_18", cls_head="mlh", num_labels=3, is_3d=True, dropout_rate=0.5
    )
    ckpt = torch.load(CHECKPOINT_R3D, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt.get('model_state_dict', ckpt))
    backbone.cls_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
    backbone.to(device)
    backbone.eval()

    # MLP definition (Must match training architecture)
    class rainMLP(nn.Module):
        def __init__(self, input_dim=513, num_classes=3):
            super(rainMLP, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
            )
        def forward(self, x): return self.network(x)

    # Load MLP and Scaler from your best checkpoint
    mlp_ckpt = torch.load(CHECKPOINT_MLP, map_location=device, weights_only=False)
    mlp_model = rainMLP()
    mlp_model.load_state_dict(mlp_ckpt['model_state_dict'])
    scaler = mlp_ckpt['scaler'] # This is the critical scaler from the 22k training
    mlp_model.to(device)
    mlp_model.eval()
    print(f"Best Model (Epoch {mlp_ckpt['epoch']}) loaded successfully.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit()

def get_dcp_score(frame):
    # Standardizing DCP to 224x224 to match the logic of the extraction script
    small_frame = cv2.resize(frame, (224, 224))
    min_channel = np.min(small_frame, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(min_channel, kernel)
    A = np.max(dark)
    t = 1 - 0.95 * (dark / (A if A > 0 else 1.0))
    return np.mean(t)

# --- 3. PROCESSING LOOP WITH SLIDING WINDOW ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

frame_buffer = []  
dcp_buffer = []
current_label = "Initializing..."
current_conf = 0.0

print(f"\n{'Time':<10} | {'DCP Avg':<10} | {'Prediction':<15} | {'Confidence'}")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Pre-process frame for R3D
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    
    frame_buffer.append(tensor)
    dcp_buffer.append(get_dcp_score(frame))

    # Trigger inference when buffer is full
    if len(frame_buffer) == WINDOW_SIZE:
        input_clip = torch.stack(frame_buffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 1. Deep Features
            deep_feats = backbone(input_clip).cpu().numpy().flatten()
            if len(deep_feats) > 512: 
                # Spatial averaging if feature map is larger than 1x1
                deep_feats = deep_feats.reshape(512, -1).mean(axis=1)

            # 2. Physics Feature
            avg_dcp = np.mean(dcp_buffer)

            # 3. Hybrid Fusion & Scaling
            combined = np.hstack([deep_feats, [avg_dcp]]).reshape(1, -1)
            scaled_input = scaler.transform(combined)
            
            # 4. MLP Prediction
            logits = mlp_model(torch.FloatTensor(scaled_input).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            pred_idx = np.argmax(probs)
            current_label = CLASS_NAMES[pred_idx]
            current_conf = probs[pred_idx] * 100

            timestamp = f"{int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)//60:02d}:{int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60:02d}"
            print(f"{timestamp:<10} | {avg_dcp:<10.3f} | {current_label:<15} | {current_conf:.1f}%")

        # SLIDING WINDOW: Remove the oldest 4 frames to make room for new ones
        # This keeps the model responsive and matches your training stride
        frame_buffer = frame_buffer[STRIDE:]
        dcp_buffer = dcp_buffer[STRIDE:]

    # Visual UI
    display_color = (0, 255, 0) if "Clear" in current_label else (0, 165, 255) if "Light" in current_label else (0, 0, 255)
    cv2.rectangle(frame, (20, 20), (350, 120), (0,0,0), -1) # Background box
    cv2.putText(frame, f"STATUS: {current_label}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, display_color, 2)
    cv2.putText(frame, f"CONF: {current_conf:.1f}%", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Hybrid Weather Intelligence", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time

# Ensure models folder is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models

# --- 1. SETTINGS & PATHS ---
# UPDATED: Live Camera RTSP Link
VIDEO_PATH = "rtsp://TAPO941A:visibility@192.168.0.112:554/stream2" 
CHECKPOINT_R3D = "r3d18_checkpoints_batch4/checkpoint_epoch_4.pth"
CHECKPOINT_MLP = "mlp_checkpoints_rigorous_v4/rigorous_epoch_28.pth" 

CLASS_NAMES = ['Clear', 'Light Rain', 'Heavy Rain']
WINDOW_SIZE = 16 
STRIDE = 4 

# --- 2. LOAD MODELS ---
device = torch.device("cpu") # Change to "cuda" if you have a GPU
print(f"Loading models on {device}...")

try:
    backbone = models.Recognizer3D(
        backbone="r3d_18", cls_head="mlh", num_labels=3, is_3d=True, dropout_rate=0.5
    )
    ckpt = torch.load(CHECKPOINT_R3D, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt.get('model_state_dict', ckpt))
    backbone.cls_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
    backbone.to(device)
    backbone.eval()

    class rainMLP(nn.Module):
        def __init__(self, input_dim=513, num_classes=3):
            super(rainMLP, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
            )
        def forward(self, x): return self.network(x)

    mlp_ckpt = torch.load(CHECKPOINT_MLP, map_location=device, weights_only=False)
    mlp_model = rainMLP()
    mlp_model.load_state_dict(mlp_ckpt['model_state_dict'])
    scaler = mlp_ckpt['scaler'] 
    mlp_model.to(device)
    mlp_model.eval()
    print(f"Model successfully loaded.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit()

def get_dcp_score(frame):
    small_frame = cv2.resize(frame, (224, 224))
    min_channel = np.min(small_frame, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(min_channel, kernel)
    A = np.max(dark)
    t = 1 - 0.95 * (dark / (A if A > 0 else 1.0))
    return np.mean(t)

# --- 3. LIVE STREAM PROCESSING ---
print(f"Connecting to: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

# Critical for RTSP: Set buffer size small to ensure we see the "latest" frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

frame_buffer = []  
dcp_buffer = []
current_label = "Initializing..."
current_conf = 0.0

print(f"\n{'Real-Time':<10} | {'DCP Avg':<10} | {'Prediction':<15} | {'Confidence'}")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        time.sleep(1)
        cap.open(VIDEO_PATH)
        continue

    # Pre-process
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    
    frame_buffer.append(tensor)
    dcp_buffer.append(get_dcp_score(frame))

    if len(frame_buffer) == WINDOW_SIZE:
        input_clip = torch.stack(frame_buffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        
        with torch.no_grad():
            deep_feats = backbone(input_clip).cpu().numpy().flatten()
            if len(deep_feats) > 512: 
                deep_feats = deep_feats.reshape(512, -1).mean(axis=1)

            avg_dcp = np.mean(dcp_buffer)
            combined = np.hstack([deep_feats, [avg_dcp]]).reshape(1, -1)
            scaled_input = scaler.transform(combined)
            
            logits = mlp_model(torch.FloatTensor(scaled_input).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            pred_idx = np.argmax(probs)
            current_label = CLASS_NAMES[pred_idx]
            current_conf = probs[pred_idx] * 100

            # Real-world timestamp
            t_now = time.strftime("%H:%M:%S")
            print(f"{t_now:<10} | {avg_dcp:<10.3f} | {current_label:<15} | {current_conf:.1f}%")

        frame_buffer = frame_buffer[STRIDE:]
        dcp_buffer = dcp_buffer[STRIDE:]

    # Visual UI
    display_color = (0, 255, 0) if "Clear" in current_label else (0, 165, 255) if "Light" in current_label else (0, 0, 255)
    cv2.rectangle(frame, (20, 20), (350, 120), (0,0,0), -1) 
    cv2.putText(frame, f"STATUS: {current_label}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, display_color, 2)
    cv2.putText(frame, f"CONF: {current_conf:.1f}%", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Scale down display for visibility
    cv2.imshow("Live Weather Intelligence", cv2.resize(frame, (960, 540)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()