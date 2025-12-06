import os
import numpy as np
import csv
import pandas as pd

def save_features(X, y, files, save_dir):
    print("[*] Saving features to file...")
    os.makedirs(save_dir, exist_ok=True)
    features_path = os.path.join(save_dir, 'features.npz')
    np.savez_compressed(features_path, X=X, y=y, files=files)
    csv_path = os.path.join(save_dir, 'features.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['filename', 'label'] + [f'feature_{i}' for i in range(X.shape[1])]
        writer.writerow(header)
        for i in range(X.shape[0]):
            row = [files[i], y[i]] + X[i].tolist()
            writer.writerow(row)
    print(f"[+] Features saved to: {features_path}")
    print(f"[+] CSV format features saved to: {csv_path}")

def save_features_to_csv(X, y, files, output_path):
    print(f"[*] Saving features to {output_path}...")
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df_data = { 'filename': files, 'label': y }
    for i, feature_name in enumerate(feature_names):
        df_data[feature_name] = X[:, i]
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    print(f"[+] Features saved to: {output_path}")

def save_features_to_pickle(X, y, files, output_path):
    print(f"[*] Saving features to {output_path}...")
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df_data = { 'filename': files, 'label': y }
    for i, feature_name in enumerate(feature_names):
        df_data[feature_name] = X[:, i]
    df = pd.DataFrame(df_data)
    df.to_pickle(output_path)
    print(f"[+] Features saved to: {output_path}")