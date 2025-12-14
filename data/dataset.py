import os
import numpy as np

class MalwareDataset:
    def __init__(self, data_dir, file_list, label_list, max_length=256*1024):
        self.data_dir = data_dir
        self.file_list = file_list
        self.label_list = label_list
        self.max_length = max_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.label_list[idx]
        if filename.endswith('.npz'):
            file_path = os.path.join(self.data_dir, filename)
        else:
            file_path = os.path.join(self.data_dir, f"{filename}.npz")
        try:
            with np.load(file_path) as data:
                byte_sequence = data['byte_sequence']
                if 'pe_features' in data:
                    pe_features = data['pe_features']
                    if pe_features.ndim > 1:
                        pe_features = pe_features.flatten()
                else:
                    from config.config import PE_FEATURE_VECTOR_DIM
                    pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                orig_length = int(data['orig_length']) if 'orig_length' in data else self.max_length
        except FileNotFoundError:
            print(f"[Warning] File not found: {file_path}, using zero padding.")
            byte_sequence = np.zeros(self.max_length, dtype=np.uint8)
            from config.config import PE_FEATURE_VECTOR_DIM
            pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
            orig_length = 0
        except Exception as e:
            print(f"[Warning] Error reading file {file_path}: {e}, using zero padding.")
            byte_sequence = np.zeros(self.max_length, dtype=np.uint8)
            from config.config import PE_FEATURE_VECTOR_DIM
            pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
            orig_length = 0
        if len(byte_sequence) > self.max_length:
            byte_sequence = byte_sequence[:self.max_length]
        else:
            byte_sequence = np.pad(byte_sequence, (0, self.max_length - len(byte_sequence)), 'constant')
        
        return byte_sequence, pe_features, label, orig_length
