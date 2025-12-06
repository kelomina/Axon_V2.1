import os
import json
import numpy as np
from tqdm import tqdm

from data.dataset import MalwareDataset
from features.statistics import extract_statistical_features
from config.config import DEFAULT_MAX_FILE_SIZE

def load_dataset(data_dir, metadata_file, max_file_size=DEFAULT_MAX_FILE_SIZE, fast_dev_run=False):
    print("[*] Loading dataset...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    label_map = {}
    for file, label in metadata.items():
        fname_lower = file.lower()
        if ('待加入白名单' in file) or ('whitelist' in fname_lower) or ('benign' in fname_lower) or ('good' in fname_lower) or ('clean' in fname_lower):
            label_map[file] = 0
        elif ('malicious' in fname_lower) or ('virus' in fname_lower) or ('trojan' in fname_lower):
            label_map[file] = 1
        elif label == 'benign' or label == 0:
            label_map[file] = 0
        elif label == 'malicious' or label == 1:
            label_map[file] = 1
        elif label == '待加入白名单':
            label_map[file] = 0
        else:
            label_map[file] = 1
    benign_count = sum(1 for v in label_map.values() if v == 0)
    malicious_count = sum(1 for v in label_map.values() if v == 1)
    print(f"[+] Label distribution: Benign={benign_count}, Malicious={malicious_count}")
    all_files = list(metadata.keys())
    all_labels = [label_map[fname] for fname in all_files]
    if fast_dev_run:
        print("[!] Fast development mode enabled, balancing benign and malicious samples.")
        benign_files = [f for f, label in zip(all_files, all_labels) if label == 0]
        malicious_files = [f for f, label in zip(all_files, all_labels) if label == 1]
        n_samples_per_class = 5000
        selected_benign_files = benign_files[:min(n_samples_per_class, len(benign_files))]
        selected_malicious_files = malicious_files[:min(n_samples_per_class, len(malicious_files))]
        all_files = selected_benign_files + selected_malicious_files
        all_labels = [0] * len(selected_benign_files) + [1] * len(selected_malicious_files)
        print(f"    Benign samples: {len(selected_benign_files)}")
        print(f"    Malicious samples: {len(selected_malicious_files)}")
    print(f"[+] Loaded {len(all_files)} files")
    features_list = []
    labels_list = []
    valid_files = []
    dataset = MalwareDataset(data_dir, all_files, all_labels, max_file_size)
    total_samples = len(dataset)
    progress_desc = "Extracting features"
    for i in tqdm(range(total_samples), desc=progress_desc):
        try:
            byte_sequence, pe_features, label, orig_length = dataset[i]
            features = extract_statistical_features(byte_sequence, pe_features, orig_length)
            features_list.append(features)
            labels_list.append(label)
            valid_files.append(all_files[i])
        except Exception as e:
            print(f"[!] Error processing file {all_files[i]}: {e}")
            continue
    try:
        X = np.array(features_list, dtype=np.float32)
    except ValueError as e:
        print(f"[!] Feature array shape inconsistency: {e}")
        print("[*] Attempting to manually align feature dimensions...")
        max_features = max(len(f) for f in features_list)
        aligned_features = []
        for f in features_list:
            if len(f) < max_features:
                padded_f = np.zeros(max_features, dtype=np.float32)
                padded_f[:len(f)] = f
                aligned_features.append(padded_f)
            else:
                aligned_features.append(f)
        X = np.array(aligned_features, dtype=np.float32)
    y = np.array(labels_list)
    print(f"[+] Feature extraction completed, feature dimension: {X.shape[1]}")
    print(f"[+] Valid samples: {X.shape[0]}")
    return X, y, valid_files

def extract_features_from_raw_files(data_dir, output_dir, max_file_size=DEFAULT_MAX_FILE_SIZE, file_extensions=None, label_inference='filename'):
    print(f"[*] Extracting features from raw files: {data_dir}")
    print(f"[*] Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_extensions:
                _, ext = os.path.splitext(file)
                if ext.lower() not in file_extensions:
                    continue
            all_files.append(file_path)
    if not all_files:
        print(f"[!] No files found in raw file directory: {data_dir}")
        return [], []
    print(f"[+] Found {len(all_files)} files in raw file directory")
    try:
        from features.extractor_save import process_file_directory
        print("[+] Successfully imported feature extraction module")
    except ImportError as e:
        print(f"[!] Failed to import feature extraction module: {e}")
        return [], []
    labels = []
    output_files = []
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, data_dir)
        output_file = os.path.join(output_dir, rel_path + '.npz')
        output_files.append(output_file)
        output_subdir = os.path.dirname(output_file)
        os.makedirs(output_subdir, exist_ok=True)
        if label_inference == 'filename':
            file_name = os.path.basename(file_path)
            if 'benign' in file_name.lower() or 'good' in file_name.lower() or 'clean' in file_name.lower():
                labels.append(0)
            else:
                labels.append(1)
        elif label_inference == 'directory':
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if 'benign' in parent_dir.lower() or 'good' in parent_dir.lower() or 'clean' in parent_dir.lower():
                labels.append(0)
            else:
                labels.append(1)
        else:
            labels.append(1)
    print("[*] Starting feature extraction...")
    success_count = 0
    from tqdm import tqdm
    for i, (input_file, output_file) in enumerate(tqdm(zip(all_files, output_files), total=len(all_files), desc="Feature extraction")):
        try:
            process_file_directory(input_file, output_file, max_file_size)
            success_count += 1
        except Exception as e:
            print(f"[!] Error processing file {input_file}: {e}")
            if output_file in output_files:
                idx = output_files.index(output_file)
                output_files.pop(idx)
                labels.pop(idx)
    print(f"[+] Feature extraction completed: {success_count}/{len(all_files)} files processed successfully")
    file_names = [os.path.relpath(f, output_dir) for f in output_files]
    return file_names, labels

def load_incremental_dataset(data_dir, max_file_size=DEFAULT_MAX_FILE_SIZE):
    print(f"[*] Loading dataset from incremental directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"[!] Incremental training directory does not exist: {data_dir}")
        return None, None, None
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                all_files.append(os.path.join(root, file))
    if not all_files:
        print(f"[!] No .npz files found in incremental training directory: {data_dir}")
        return None, None, None
    print(f"[+] Found {len(all_files)} files in incremental directory")
    labels = []
    valid_files = []
    file_names = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        if 'benign' in file_name.lower() or 'good' in file_name.lower() or 'clean' in file_name.lower():
            labels.append(0)
        else:
            labels.append(1)
        valid_files.append(file_path)
        file_names.append(file_name)
    features_list = []
    valid_file_names = []
    for i, file_path in enumerate(tqdm(valid_files, desc="Extracting incremental features")):
        try:
            with np.load(file_path) as data:
                byte_sequence = data['byte_sequence']
                if 'pe_features' in data:
                    pe_features = data['pe_features']
                    if pe_features.ndim > 1:
                        pe_features = pe_features.flatten()
                else:
                    pe_features = np.zeros(1000, dtype=np.float32)
                orig_length = data['orig_length'] if 'orig_length' in data else max_file_size
            if len(byte_sequence) > max_file_size:
                byte_sequence = byte_sequence[:max_file_size]
            else:
                byte_sequence = np.pad(byte_sequence, (0, max_file_size - len(byte_sequence)), 'constant')
            if len(pe_features) != 1000:
                fixed_pe_features = np.zeros(1000, dtype=np.float32)
                fixed_pe_features[:min(len(pe_features), 1000)] = pe_features[:min(len(pe_features), 1000)]
                pe_features = fixed_pe_features
            features = extract_statistical_features(byte_sequence, pe_features, int(orig_length))
            features_list.append(features)
            valid_file_names.append(file_names[i])
        except Exception as e:
            print(f"[!] Error processing file {file_path}: {e}")
            continue
    if not features_list:
        print("[!] Failed to extract any features from incremental data")
        return None, None, None
    try:
        X = np.array(features_list, dtype=np.float32)
    except ValueError as e:
        print(f"[!] Incremental feature array shape inconsistency: {e}")
        return None, None, None
    y = np.array(labels[:len(features_list)])
    print(f"[+] Incremental feature extraction completed, feature dimension: {X.shape[1]}")
    print(f"[+] Valid samples: {X.shape[0]}")
    return X, y, valid_file_names
