import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config.config import (
    GATING_MODE, GATING_INPUT_DIM, GATING_HIDDEN_DIM, GATING_OUTPUT_DIM,
    GATING_LEARNING_RATE, GATING_EPOCHS, GATING_BATCH_SIZE, GATING_MODEL_PATH,
    EXPERT_NORMAL_MODEL_PATH, EXPERT_PACKED_MODEL_PATH,
    FEATURES_PKL_PATH, PROCESSED_DATA_DIR, METADATA_FILE, DEFAULT_MAX_FILE_SIZE,
    DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
)
from models.gating import create_gating_model
from training.train_lightgbm import train_lightgbm_model
from training.data_loader import load_dataset

# Feature indices based on analysis
# Statistical features (49) + Lightweight PE (256) + Index in PE_FEATURE_ORDER
IDX_PACKED_SECTIONS_RATIO = 49 + 256 + 18
IDX_PACKER_KEYWORD_HITS_COUNT = 49 + 256 + 107

def generate_routing_labels(X):
    """
    Generate routing labels based on heuristics.
    Label 1 (Packed): High packed section ratio or packer keyword hits.
    Label 0 (Normal): Otherwise.
    """
    print("[*] Generating routing labels based on heuristics...")
    
    # Check feature dimension
    if X.shape[1] <= max(IDX_PACKED_SECTIONS_RATIO, IDX_PACKER_KEYWORD_HITS_COUNT):
        print(f"[!] Warning: Feature dimension {X.shape[1]} is smaller than expected indices.")
        # Fallback to zeros if indices are out of bounds (should not happen if consistent)
        return np.zeros(len(X), dtype=int)

    packed_ratio = X[:, IDX_PACKED_SECTIONS_RATIO]
    packer_hits = X[:, IDX_PACKER_KEYWORD_HITS_COUNT]
    
    # Heuristic: Packed if packed_sections_ratio > 0.4 OR packer_keyword_hits_count > 0
    is_packed = (packed_ratio > 0.4) | (packer_hits > 0)
    
    labels = is_packed.astype(int)
    print(f"    Total samples: {len(labels)}")
    print(f"    Normal samples: {np.sum(labels == 0)}")
    print(f"    Packed samples: {np.sum(labels == 1)}")
    return labels

def train_gating_model_process(X_train, y_train, X_val, y_val):
    print(f"[*] Training Gating Model ({GATING_MODE})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    model = create_gating_model(GATING_MODE, GATING_INPUT_DIM, GATING_HIDDEN_DIM, GATING_OUTPUT_DIM)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=GATING_LEARNING_RATE)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=GATING_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=GATING_BATCH_SIZE)
    
    best_val_acc = 0.0
    
    for epoch in range(GATING_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"    Epoch {epoch+1}/{GATING_EPOCHS} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), GATING_MODEL_PATH)
            
    print(f"[+] Gating Model saved to {GATING_MODEL_PATH} (Best Acc: {best_val_acc:.4f})")
    return model

def main():
    # 1. Load Data
    if os.path.exists(FEATURES_PKL_PATH):
        print(f"[*] Loading features from {FEATURES_PKL_PATH}...")
        import pandas as pd
        df = pd.read_pickle(FEATURES_PKL_PATH)
        X = df.drop(['filename', 'label'], axis=1).values
        y = df['label'].values
        files = df['filename'].tolist()
    else:
        print(f"[*] Extracting features (this may take a while)...")
        # Assuming load_dataset returns (X, y, files) or similar
        # Based on previous read of pretrain.py, load_dataset returns X, y, files
        X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, DEFAULT_MAX_FILE_SIZE)
        # Note: If features need to be saved, we should do it here, but for now we proceed.

    print(f"[*] Total samples: {len(X)}")
    print(f"[*] Feature dimension: {X.shape[1]}")
    
    # 2. Generate Routing Labels
    routing_labels = generate_routing_labels(X)
    
    # 3. Train Gating Model
    # Split for Gating Model Training
    X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(
        X, routing_labels, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=routing_labels
    )
    
    train_gating_model_process(X_train_g, y_train_g, X_val_g, y_val_g)
    
    # 4. Train Expert Models
    print("\n[*] Training Expert Models...")
    
    # Split Data by Routing Label (using ALL data, or train/test split? 
    # Usually we want to train experts on the training set of the main task.
    # Let's split the main dataset first into Train/Test for the overall task evaluation.
    
    X_train_main, X_test_main, y_train_main, y_test_main, r_train_main, r_test_main = train_test_split(
        X, y, routing_labels, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y
    )
    
    # Expert Normal (Routing Label 0)
    # We use samples from X_train_main where r_train_main == 0
    mask_normal = (r_train_main == 0)
    X_normal = X_train_main[mask_normal]
    y_normal = y_train_main[mask_normal]
    
    # We need a validation set for LightGBM
    if len(X_normal) > 10:
        X_t_norm, X_v_norm, y_t_norm, y_v_norm = train_test_split(
            X_normal, y_normal, test_size=0.1, random_state=DEFAULT_RANDOM_STATE
        )
        print(f"[*] Expert Normal - Train: {len(X_t_norm)}, Val: {len(X_v_norm)}")
        model_normal = train_lightgbm_model(X_t_norm, y_t_norm, X_v_norm, y_v_norm)
        model_normal.save_model(EXPERT_NORMAL_MODEL_PATH)
        print(f"[+] Expert Normal saved to {EXPERT_NORMAL_MODEL_PATH}")
    else:
        print("[!] Not enough samples for Expert Normal training.")

    # Expert Packed (Routing Label 1)
    mask_packed = (r_train_main == 1)
    X_packed = X_train_main[mask_packed]
    y_packed = y_train_main[mask_packed]
    
    if len(X_packed) > 10:
        X_t_pack, X_v_pack, y_t_pack, y_v_pack = train_test_split(
            X_packed, y_packed, test_size=0.1, random_state=DEFAULT_RANDOM_STATE
        )
        print(f"[*] Expert Packed - Train: {len(X_t_pack)}, Val: {len(X_v_pack)}")
        model_packed = train_lightgbm_model(X_t_pack, y_t_pack, X_v_pack, y_v_pack)
        model_packed.save_model(EXPERT_PACKED_MODEL_PATH)
        print(f"[+] Expert Packed saved to {EXPERT_PACKED_MODEL_PATH}")
    else:
        print("[!] Not enough samples for Expert Packed training.")

    print("\n[*] Training pipeline completed.")

if __name__ == '__main__':
    main()
