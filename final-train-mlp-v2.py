import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# --- 1. SETTINGS ---
csv_path = "no_dcp_features.csv" 
checkpoint_dir = "mlp_checkpoints_rigorous_v4.pth"
history_path = "no_dcp_training_history.csv"
batch_size = 32
learning_rate = 0.001
epochs = 50
os.makedirs(checkpoint_dir, exist_ok=True)

# --- 2. DATA LOADING & RIGOROUS SEQUENTIAL SPLIT ---
print("Reading CSV and performing Per-Class Sequential Split...")
df = pd.read_csv(csv_path)

train_data = []
val_data = []

# Loop through each weather class to split them individually
for label in [0.0, 1.0, 2.0]:
    class_df = df[df['label'] == label]
    if len(class_df) == 0: continue
    
    # RIGOROUS SPLIT: 80% start of sequence for train, 20% end for validation
    split_idx = int(len(class_df) * 0.8)
    train_data.append(class_df.iloc[:split_idx])
    val_data.append(class_df.iloc[split_idx:])

# Combine and shuffle ONLY the training set
train_df = pd.concat(train_data).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat(val_data)

X_train_raw = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_val_raw = val_df.drop('label', axis=1).values
y_val = val_df['label'].values

# --- 3. SCALING & DATALOADERS ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)

# --- 4. MLP MODEL ---
class rainMLP(nn.Module):
    def __init__(self, input_dim=513, num_classes=3):
        super(rainMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)

device = torch.device("cpu") # Change to "cuda" if GPU is available
model = rainMLP(input_dim=X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# --- 5. TRAINING LOOP ---
history = []
best_val_acc = 0.0

header = f"{'Epoch':<6} | {'Tr. Loss':<10} | {'Val Loss':<10} | {'Acc %':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}"
print("-" * len(header))
print(header)
print("-" * len(header))

for epoch in range(1, epochs + 1):
    # Training Phase
    model.train()
    train_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # Validation Phase
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            val_loss += criterion(outputs, by).item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(by.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    
    # Calculate Metrics
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    # Print Table Row
    print(f"{epoch:<6} | {avg_train_loss:<10.4f} | {avg_val_loss:<10.4f} | {accuracy:<8.2f} | {precision:<8.4f} | {recall:<8.4f} | {f1:<8.4f}")

    # Track History
    history.append([epoch, avg_train_loss, avg_val_loss, accuracy, precision, recall, f1])

    # Save Checkpoint
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'metrics': {'accuracy': accuracy, 'f1': f1, 'val_loss': avg_val_loss}
    }, f"{checkpoint_dir}/rigorous_epoch_{epoch}.pth")

# --- 6. EXPORT ---
history_df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'val_loss', 'accuracy', 'precision', 'recall', 'f1'])
history_df.to_csv(history_path, index=False)
print("-" * len(header))
print(f"Rigorous Training Complete. History saved to {history_path}")