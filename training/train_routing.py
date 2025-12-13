import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score

from config.config import (
    GATING_MODE, GATING_INPUT_DIM, GATING_HIDDEN_DIM, GATING_OUTPUT_DIM,
    GATING_LEARNING_RATE, GATING_EPOCHS, GATING_BATCH_SIZE, GATING_MODEL_PATH,
    EXPERT_NORMAL_MODEL_PATH, EXPERT_PACKED_MODEL_PATH,
    FEATURES_PKL_PATH, PROCESSED_DATA_DIR, METADATA_FILE, DEFAULT_MAX_FILE_SIZE,
    DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE,
    ROUTING_EVAL_REPORT_PATH, ROUTING_CONFUSION_MATRIX_PATH, MODEL_EVAL_FIG_DIR,
    EVAL_FONT_FAMILY, PREDICTION_THRESHOLD,
    EVAL_TOP_FEATURE_COUNT,
    PACKED_SECTIONS_RATIO_THRESHOLD, PACKER_KEYWORD_HITS_THRESHOLD,
    DEFAULT_MAX_FINETUNE_ITERATIONS
)
from models.gating import create_gating_model
from training.train_lightgbm import train_lightgbm_model
from training.data_loader import load_dataset, extract_features_from_raw_files, load_incremental_dataset
from training.feature_io import save_features_to_pickle
from training.model_io import load_existing_model, save_model
from training.evaluate import evaluate_model
from features.extractor_in_memory import PE_FEATURE_ORDER

from models.routing_model import RoutingModel

# Feature indices based on analysis
STAT_FEATURE_DIM = 49 
LIGHTWEIGHT_PE_DIM = 256
IDX_PACKED_SECTIONS_RATIO = STAT_FEATURE_DIM + LIGHTWEIGHT_PE_DIM + PE_FEATURE_ORDER.index('packed_sections_ratio')
IDX_PACKER_KEYWORD_HITS_COUNT = STAT_FEATURE_DIM + LIGHTWEIGHT_PE_DIM + PE_FEATURE_ORDER.index('packer_keyword_hits_count')

def get_feature_semantics(index):
    n_stat = 49
    if index < n_stat:
        if index == 0: return '字节均值'
        elif index == 1: return '字节标准差'
        elif index == 2: return '字节最小值'
        elif index == 3: return '字节最大值'
        elif index == 4: return '字节中位数'
        elif index == 5: return '字节25分位'
        elif index == 6: return '字节75分位'
        elif index == 7: return '零字节计数'
        elif index == 8: return '0xFF字节计数'
        elif index == 9: return '0x90字节计数'
        elif index == 10: return '可打印字节计数'
        elif index == 11: return '全局熵'
        elif 12 <= index <= 20:
            pos = (index - 12) // 3
            mod = (index - 12) % 3
            seg = ['前段','中段','后段'][pos]
            name = ['均值','标准差','熵'][mod]
            return seg + name
        elif 21 <= index <= 30: return f'分块均值_{index-21}'
        elif 31 <= index <= 40: return f'分块标准差_{index-31}'
        elif 41 <= index <= 44: return ['分块均值差绝对均值','分块均值差标准差','分块均值差最大值','分块均值差最小值'][index-41]
        elif 45 <= index <= 48: return ['分块标准差差绝对均值','分块标准差差标准差','分块标准差差最大值','分块标准差差最小值'][index-45]
        else: return '统计特征'
    j = index - n_stat
    if j < 256:
        if j < 128: return '轻量哈希位:导入DLL'
        elif j < 224: return '轻量哈希位:导入API'
        else: return '轻量哈希位:节名'
    k = j - 256
    if k < len(PE_FEATURE_ORDER):
        return PE_FEATURE_ORDER[k]
    return 'PE特征'

def evaluate_routing_system(X_test, y_test, files_test=None):
    print("\n[*] Evaluating Routing System on Test Set...")
    
    # Reload the full system
    try:
        routing_model = RoutingModel(device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"[!] Failed to load Routing System for evaluation: {e}")
        return

    # Predictions
    print("    Running predictions...")
    predictions, routing_decisions = routing_model.predict(X_test)
    
    # Binary Classification Metrics
    y_pred_binary = (predictions > PREDICTION_THRESHOLD).astype(int)
    
    acc = accuracy_score(y_test, y_pred_binary)
    print(f"[+] System Accuracy: {acc:.4f}")
    
    report = classification_report(y_test, y_pred_binary, target_names=['Benign', 'Malicious'])
    print("\n[*] Classification Report:")
    print(report)
    
    # Routing Stats
    stats = routing_model.get_routing_stats(routing_decisions)
    print("\n[*] Routing Statistics on Test Set:")
    print(f"    Total: {stats['total']}")
    print(f"    Routed to Normal Expert: {stats['normal']} ({stats['normal']/stats['total']:.1%})")
    print(f"    Routed to Packed Expert: {stats['packed']} ({stats['packed_ratio']:.1%})")

    # Threshold Sensitivity Analysis
    print("\n[*] Threshold sensitivity (0.90–0.99):")
    thresholds = np.arange(0.90, 1.00, 0.01)
    for t in thresholds:
        y_pred_t = (predictions > t).astype(int)
        cm_t = confusion_matrix(y_test, y_pred_t)
        if cm_t.shape == (2, 2):
            tn, fp, fn, tp = cm_t.ravel()
        else:
            tn = fp = fn = tp = 0 # Handle edge cases
            if len(np.unique(y_test)) == 1:
                if y_test[0] == 0: tn = len(y_test)
                else: tp = len(y_test) # Rough approx if prediction matches

        acc_t = accuracy_score(y_test, y_pred_t)
        pre_t = precision_score(y_test, y_pred_t, zero_division=0)
        rec_t = recall_score(y_test, y_pred_t, zero_division=0)
        fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"    t={t:.2f} acc={acc_t:.4f} pre={pre_t:.4f} rec={rec_t:.4f} FPR={fpr_t:.4f} TPR={tpr_t:.4f} FP={int(fp)}")

    # Feature Importance (from Expert Normal as proxy for general importance)
    if routing_model.expert_normal and routing_model.expert_normal.model:
        print(f"\n[*] Top {EVAL_TOP_FEATURE_COUNT} important features (Expert Normal):")
        feature_importance = routing_model.expert_normal.model.feature_importance(importance_type='gain')
        indices_sorted = np.argsort(feature_importance)[::-1]
        for rank, idx in enumerate(indices_sorted[:EVAL_TOP_FEATURE_COUNT], 1):
            semantics = get_feature_semantics(idx)
            print(f"    {rank:2d}. feature_{idx}: {feature_importance[idx]:.2f} ({semantics})")

    # --- Reporting & Visualization ---
    os.makedirs(MODEL_EVAL_FIG_DIR, exist_ok=True)

    # 1. Save Text Report
    with open(ROUTING_EVAL_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("Routing System Evaluation Report\n")
        f.write("================================\n\n")
        f.write(f"System Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nRouting Statistics:\n")
        f.write(f"    Total: {stats['total']}\n")
        f.write(f"    Routed to Normal Expert: {stats['normal']} ({stats['normal']/stats['total']:.1%})\n")
        f.write(f"    Routed to Packed Expert: {stats['packed']} ({stats['packed_ratio']:.1%})\n")
    print(f"[+] Evaluation report saved to {ROUTING_EVAL_REPORT_PATH}")

    # 2. Plot Confusion Matrix
    try:
        cm = confusion_matrix(y_test, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malicious'], 
                    yticklabels=['Benign', 'Malicious'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Routing System Confusion Matrix')
        plt.tight_layout()
        plt.savefig(ROUTING_CONFUSION_MATRIX_PATH)
        plt.close()
        print(f"[+] Confusion matrix plot saved to {ROUTING_CONFUSION_MATRIX_PATH}")
    except Exception as e:
        print(f"[!] Failed to generate confusion matrix plot: {e}")

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
        return np.zeros(len(X), dtype=int)

    packed_ratio = X[:, IDX_PACKED_SECTIONS_RATIO]
    packer_hits = X[:, IDX_PACKER_KEYWORD_HITS_COUNT]
    
    # Heuristic: Packed if packed_sections_ratio > PACKED_SECTIONS_RATIO_THRESHOLD OR packer_keyword_hits_count > PACKER_KEYWORD_HITS_THRESHOLD
    is_packed = (packed_ratio > PACKED_SECTIONS_RATIO_THRESHOLD) | (packer_hits > PACKER_KEYWORD_HITS_THRESHOLD)
    
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

def train_expert_model_with_finetuning(X_train, y_train, X_val, y_val, files_train, files_val, 
                                     model_path, args, expert_name="Expert"):
    """
    Train an expert model with optional incremental training and False Positive finetuning.
    """
    print(f"\n[*] Training {expert_name}...")
    
    existing_model = None
    if args.incremental_training and os.path.exists(model_path):
        print(f"    Loading existing model for incremental training: {model_path}")
        existing_model = load_existing_model(model_path)

    model = None
    if existing_model:
        model = train_lightgbm_model(
            X_train, y_train, X_val, y_val, 
            files_train=files_train,
            num_boost_round=args.incremental_rounds,
            init_model=existing_model
        )
    else:
        model = train_lightgbm_model(
            X_train, y_train, X_val, y_val,
            files_train=files_train,
            num_boost_round=args.num_boost_round
        )

    # False Positive Finetuning
    if args.finetune_on_false_positives:
        print(f"[*] Starting False Positive Finetuning for {expert_name}...")
        
        # We need to evaluate on a hold-out set to find FPs. 
        # Here we use X_val as a proxy if we don't have a separate test set passed in.
        # But wait, we should really use the X_val FPs to improve training? 
        # Standard practice: Use Validation FPs to hard-mine.
        
        current_X_train = X_train
        current_y_train = y_train
        current_files_train = files_train

        max_targeted_iterations = DEFAULT_MAX_FINETUNE_ITERATIONS
        for i in range(max_targeted_iterations):
            # Evaluate on Validation Set
            # Note: We use X_val to find FPs, then add them to Train.
            # This "leaks" Val into Train, but for FP mining it's often accepted or we need a 3rd split.
            # Here we follow the aggressive approach.
            
            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > PREDICTION_THRESHOLD).astype(int)
            
            # Find FPs
            fp_indices = np.where((y_val == 0) & (y_pred == 1))[0]
            if len(fp_indices) == 0:
                print(f"    [Round {i+1}] No False Positives found in validation set.")
                break
                
            print(f"    [Round {i+1}] Found {len(fp_indices)} False Positives. Retraining...")
            
            # Extract FP samples
            X_fps = X_val[fp_indices]
            y_fps = y_val[fp_indices]
            files_fps = [files_val[idx] for idx in fp_indices]
            
            # Add to Training Data (Augmentation)
            # We assume these are "hard" negatives.
            current_X_train = np.vstack([current_X_train, X_fps])
            current_y_train = np.concatenate([current_y_train, y_fps])
            current_files_train = current_files_train + files_fps
            
            # Retrain (Incremental/Continued)
            model = train_lightgbm_model(
                current_X_train, current_y_train, X_val, y_val,
                files_train=current_files_train,
                false_positive_files=files_fps, # Highlight these specific files for weighting
                num_boost_round=args.num_boost_round,
                init_model=model,
                iteration=i+2
            )
            
    model.save_model(model_path)
    print(f"[+] {expert_name} saved to {model_path}")
    return model

def main(args=None):
    # Default Args Handling if None (for direct script execution compatibility)
    if args is None:
        parser = argparse.ArgumentParser()
        # Add minimal defaults or just rely on config
        # But really this function expects args object.
        pass

    # 1. Load Data
    use_existing = args.use_existing_features if args else False
    save_features_flag = args.save_features if args else False
    
    X, y, files = None, None, None

    # Incremental Data Loading
    if args and args.incremental_training and args.incremental_data_dir:
        if args.incremental_raw_data_dir:
            print("[*] Extracting features from raw files (Incremental)...")
            extract_features_from_raw_files(
                args.incremental_raw_data_dir,
                args.incremental_data_dir,
                args.max_file_size,
                args.file_extensions,
                args.label_inference
            )
        
        print("[*] Loading incremental dataset...")
        X, y, files = load_incremental_dataset(args.incremental_data_dir, args.max_file_size)
        if X is None:
            print("[!] Failed to load incremental data.")
            return
    
    # Standard Data Loading
    elif use_existing and os.path.exists(FEATURES_PKL_PATH):
        print(f"[*] Loading features from {FEATURES_PKL_PATH}...")
        try:
            import pandas as pd
            df = pd.read_pickle(FEATURES_PKL_PATH)
            X = df.drop(['filename', 'label'], axis=1).values
            y = df['label'].values
            files = df['filename'].tolist()
        except Exception as e:
            print(f"[!] Failed to load existing features: {e}")
            print("    Falling back to feature extraction...")
            
    if X is None:
        print(f"[*] Extracting features (this may take a while)...")
        X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, DEFAULT_MAX_FILE_SIZE, fast_dev_run=args.fast_dev_run if args else False)
        
        if save_features_flag:
            save_features_to_pickle(X, y, files, FEATURES_PKL_PATH)

    print(f"[*] Total samples: {len(X)}")
    print(f"[*] Feature dimension: {X.shape[1]}")
    
    # 2. Generate Routing Labels
    routing_labels = generate_routing_labels(X)
    
    # 3. Train Gating Model
    # Split for Gating Model Training
    X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(
        X, routing_labels, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=routing_labels
    )
    
    # Only train gating model if NOT in incremental mode or if explicitly requested?
    # For now, we always retrain gating model to ensure it adapts to new data distribution if any.
    train_gating_model_process(X_train_g, y_train_g, X_val_g, y_val_g)
    
    # 4. Train Expert Models
    print("\n[*] Training Expert Models...")
    
    # Split Data by Routing Label
    # We split the entire dataset into Train/Test first to have a global evaluation set.
    
    X_train_main, X_test_main, y_train_main, y_test_main, r_train_main, r_test_main, files_train_main, files_test_main = train_test_split(
        X, y, routing_labels, files, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y
    )
    
    # --- Expert Normal (Routing Label 0) ---
    mask_normal = (r_train_main == 0)
    X_normal = X_train_main[mask_normal]
    y_normal = y_train_main[mask_normal]
    files_normal = [files_train_main[i] for i in range(len(files_train_main)) if mask_normal[i]]
    
    if len(X_normal) > 10:
        X_t_norm, X_v_norm, y_t_norm, y_v_norm, f_t_norm, f_v_norm = train_test_split(
            X_normal, y_normal, files_normal, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE
        )
        print(f"[*] Expert Normal - Train: {len(X_t_norm)}, Val: {len(X_v_norm)}")
        
        train_expert_model_with_finetuning(
            X_t_norm, y_t_norm, X_v_norm, y_v_norm, f_t_norm, f_v_norm,
            EXPERT_NORMAL_MODEL_PATH, args, expert_name="Expert Normal"
        )
    else:
        print("[!] Not enough samples for Expert Normal training.")

    # --- Expert Packed (Routing Label 1) ---
    mask_packed = (r_train_main == 1)
    X_packed = X_train_main[mask_packed]
    y_packed = y_train_main[mask_packed]
    files_packed = [files_train_main[i] for i in range(len(files_train_main)) if mask_packed[i]]
    
    if len(X_packed) > 10:
        X_t_pack, X_v_pack, y_t_pack, y_v_pack, f_t_pack, f_v_pack = train_test_split(
            X_packed, y_packed, files_packed, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE
        )
        print(f"[*] Expert Packed - Train: {len(X_t_pack)}, Val: {len(X_v_pack)}")
        
        train_expert_model_with_finetuning(
            X_t_pack, y_t_pack, X_v_pack, y_v_pack, f_t_pack, f_v_pack,
            EXPERT_PACKED_MODEL_PATH, args, expert_name="Expert Packed"
        )
    else:
        print("[!] Not enough samples for Expert Packed training.")

    print("\n[*] Training pipeline completed.")

    # 5. Final System Evaluation
    if len(X_test_main) > 0:
        evaluate_routing_system(X_test_main, y_test_main, files_test_main)
    else:
        print("[!] No test samples available for evaluation.")

if __name__ == '__main__':
    # Argument Parsing if run directly
    from config.config import (
        DEFAULT_NUM_BOOST_ROUND, DEFAULT_INCREMENTAL_ROUNDS, DEFAULT_INCREMENTAL_EARLY_STOPPING, 
        DEFAULT_MAX_FINETUNE_ITERATIONS
    )
    
    parser = argparse.ArgumentParser(description="KoloVirusDetector Routing System Training")
    parser.add_argument('--use-existing-features', action='store_true')
    parser.add_argument('--save-features', action='store_true')
    parser.add_argument('--fast-dev-run', action='store_true')
    parser.add_argument('--finetune-on-false-positives', action='store_true')
    parser.add_argument('--incremental-training', action='store_true')
    parser.add_argument('--incremental-data-dir', type=str)
    parser.add_argument('--incremental-raw-data-dir', type=str)
    parser.add_argument('--file-extensions', type=str, nargs='+')
    parser.add_argument('--label-inference', type=str, default='filename')
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument('--incremental-rounds', type=int, default=DEFAULT_INCREMENTAL_ROUNDS)
    parser.add_argument('--incremental-early-stopping', type=int, default=DEFAULT_INCREMENTAL_EARLY_STOPPING)
    parser.add_argument('--max-finetune-iterations', type=int, default=DEFAULT_MAX_FINETUNE_ITERATIONS)
    parser.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE)

    args = parser.parse_args()
    main(args)
