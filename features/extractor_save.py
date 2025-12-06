import numpy as np
from features.extractor_in_memory import extract_features_in_memory
from config.config import DEFAULT_MAX_FILE_SIZE

def process_file_directory(input_file_path, output_file_path, max_file_size=DEFAULT_MAX_FILE_SIZE):
    byte_sequence, pe_features = extract_features_in_memory(input_file_path, max_file_size)
    if byte_sequence is None or pe_features is None:
        raise Exception(f"Failed to process file {input_file_path}")
    np.savez_compressed(output_file_path, byte_sequence=byte_sequence, pe_features=pe_features)