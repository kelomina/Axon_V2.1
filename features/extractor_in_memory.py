import os
import numpy as np
import pefile
import hashlib
from utils.path_utils import validate_path
from config.config import DEFAULT_MAX_FILE_SIZE, ENTROPY_BLOCK_SIZE, ENTROPY_SAMPLE_SIZE

def calculate_byte_entropy(byte_sequence, block_size=ENTROPY_BLOCK_SIZE):
    if byte_sequence is None or len(byte_sequence) == 0:
        return 0, 0, 0, [], 0
    hist = np.bincount(byte_sequence, minlength=256)
    prob = hist / len(byte_sequence)
    prob = prob[prob > 0]
    overall_entropy = -np.sum(prob * np.log2(prob)) / 8
    block_entropies = []
    num_blocks = min(10, max(1, len(byte_sequence) // block_size))
    if num_blocks > 1:
        block_size = len(byte_sequence) // num_blocks
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size if i < num_blocks - 1 else len(byte_sequence)
            block = byte_sequence[start_idx:end_idx]
            if len(block) > 0:
                block_hist = np.bincount(block, minlength=256)
                block_prob = block_hist / len(block)
                block_prob = block_prob[block_prob > 0]
                if len(block_prob) > 0:
                    block_entropy = -np.sum(block_prob * np.log2(block_prob)) / 8
                    block_entropies.append(block_entropy)
    else:
        block = byte_sequence
        if len(block) > 0:
            block_hist = np.bincount(block, minlength=256)
            block_prob = block_hist / len(block)
            block_prob = block_prob[block_prob > 0]
            if len(block_prob) > 0:
                block_entropy = -np.sum(block_prob * np.log2(block_prob)) / 8
                block_entropies.append(block_entropy)
    if block_entropies:
        return overall_entropy, np.min(block_entropies), np.max(block_entropies), block_entropies, np.std(block_entropies)
    else:
        return overall_entropy, overall_entropy, overall_entropy, [], 0

def extract_byte_sequence(file_path, max_file_size):
    valid_path = validate_path(file_path)
    if not valid_path:
        return None, 0
    try:
        with open(valid_path, 'rb') as f:
            f.seek(8)
            raw_bytes = np.fromfile(f, dtype=np.uint8, count=max_file_size - 8)
        orig_len = len(raw_bytes)
        if orig_len < max_file_size - 8:
            padded_sequence = np.zeros(max_file_size, dtype=np.uint8)
            padded_sequence[:orig_len] = raw_bytes
            return padded_sequence, orig_len
        full_sequence = np.zeros(max_file_size, dtype=np.uint8)
        full_sequence[:orig_len] = raw_bytes
        return full_sequence, orig_len
    except Exception:
        return None, 0

def extract_file_attributes(file_path):
    features = {}
    missing_flags = {}
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            raise ValueError
        stat = os.stat(valid_path)
        features['size'] = stat.st_size
        features['log_size'] = np.log(stat.st_size + 1)
        with open(valid_path, 'rb') as f:
            sample_data = np.fromfile(f, dtype=np.uint8, count=ENTROPY_SAMPLE_SIZE)
        avg_entropy, min_entropy, max_entropy, block_entropies, entropy_std = calculate_byte_entropy(sample_data)
        features['file_entropy_avg'] = avg_entropy
        features['file_entropy_min'] = min_entropy
        features['file_entropy_max'] = max_entropy
        features['file_entropy_range'] = max_entropy - min_entropy
        features['file_entropy_std'] = entropy_std
        if block_entropies:
            features['file_entropy_q25'] = np.percentile(block_entropies, 25)
            features['file_entropy_q75'] = np.percentile(block_entropies, 75)
            features['file_entropy_median'] = np.median(block_entropies)
            high_entropy_count = sum(1 for e in block_entropies if e > 0.8)
            features['high_entropy_ratio'] = high_entropy_count / len(block_entropies)
            low_entropy_count = sum(1 for e in block_entropies if e < 0.2)
            features['low_entropy_ratio'] = low_entropy_count / len(block_entropies)
            if len(block_entropies) > 1:
                entropy_changes = np.diff(block_entropies)
                features['entropy_change_rate'] = np.mean(np.abs(entropy_changes))
                features['entropy_change_std'] = np.std(entropy_changes)
            else:
                features['entropy_change_rate'] = 0
                features['entropy_change_std'] = 0
        else:
            features['file_entropy_q25'] = 0
            features['file_entropy_q75'] = 0
            features['file_entropy_median'] = 0
            features['high_entropy_ratio'] = 0
            features['low_entropy_ratio'] = 0
            features['entropy_change_rate'] = 0
            features['entropy_change_std'] = 0
        if len(sample_data) > 0:
            zero_ratio = np.sum(sample_data == 0) / len(sample_data)
            printable_ratio = np.sum((sample_data >= 32) & (sample_data <= 126)) / len(sample_data)
            features['zero_byte_ratio'] = zero_ratio
            features['printable_byte_ratio'] = printable_ratio
        else:
            features['zero_byte_ratio'] = 0
            features['printable_byte_ratio'] = 0
    except Exception:
        for name in ['size','log_size','file_entropy_avg','file_entropy_min','file_entropy_max','file_entropy_range','file_entropy_std','file_entropy_q25','file_entropy_q75','file_entropy_median','high_entropy_ratio','low_entropy_ratio','entropy_change_rate','entropy_change_std','zero_byte_ratio','printable_byte_ratio']:
            features[name] = 0
    return features

def extract_enhanced_pe_features(file_path):
    features = {}
    missing_flags = {}
    file_size = 0
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            raise ValueError
        pe = pefile.PE(valid_path, fast_load=True)
        try:
            with open(valid_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
        except Exception:
            file_size = 0
        features['sections_count'] = len(pe.sections) if hasattr(pe, 'sections') else 0
        features['symbols_count'] = len(pe.SYMBOL_TABLE) if hasattr(pe, 'SYMBOL_TABLE') else 0
        features['imports_count'] = 0
        features['exports_count'] = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            imports = []
            api_names = []
            dll_names = []
            features['imports_count'] = len(pe.DIRECTORY_ENTRY_IMPORT)
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8').lower() if entry.dll else ''
                dll_names.append(dll_name)
                for imp in entry.imports:
                    if imp.name:
                        func_name = imp.name.decode('utf-8')
                        imports.append((dll_name, func_name))
                        api_names.append(func_name)
            features['unique_imports'] = len(set(imports))
            features['unique_dlls'] = len(set(dll_names))
            features['unique_apis'] = len(set(api_names))
            if dll_names:
                dll_name_lengths = [len(name) for name in dll_names if name]
                features['dll_name_avg_length'] = np.mean(dll_name_lengths)
                features['dll_name_max_length'] = np.max(dll_name_lengths)
                features['dll_name_min_length'] = np.min(dll_name_lengths)
            system_dlls = {'kernel32','user32','gdi32','advapi32','shell32','ole32','comctl32'}
            imported_system_dlls = set(dll.split('.')[0].lower() for dll in dll_names if dll) & system_dlls
            features['imported_system_dlls_count'] = len(imported_system_dlls)
        else:
            features['unique_imports'] = 0
            features['unique_dlls'] = 0
            features['unique_apis'] = 0
            features['dll_name_avg_length'] = 0
            features['dll_name_max_length'] = 0
            features['dll_name_min_length'] = 0
            features['imported_system_dlls_count'] = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            features['exports_count'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
            export_names = []
            for symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if symbol.name:
                    export_names.append(symbol.name.decode('utf-8'))
            if export_names:
                export_name_lengths = [len(name) for name in export_names]
                features['export_name_avg_length'] = np.mean(export_name_lengths)
                features['export_name_max_length'] = np.max(export_name_lengths)
                features['export_name_min_length'] = np.min(export_name_lengths)
                features['exports_density'] = len(export_names) / (file_size + 1)
            else:
                features['export_name_avg_length'] = 0
                features['export_name_max_length'] = 0
                features['export_name_min_length'] = 0
                features['exports_density'] = 0
        else:
            features['exports_count'] = 0
            features['export_name_avg_length'] = 0
            features['export_name_max_length'] = 0
            features['export_name_min_length'] = 0
            features['exports_density'] = 0
        if hasattr(pe, 'sections'):
            section_names = []
            section_sizes = []
            section_vsizes = []
            code_section_size = 0
            data_section_size = 0
            code_section_vsize = 0
            data_section_vsize = 0
            executable_sections_count = 0
            writable_sections_count = 0
            readable_sections_count = 0
            rwx_sections_count = 0
            common_executable_section_names = {'.text','text','.code'}
            for section in pe.sections:
                try:
                    name = section.Name.decode('utf-8', 'ignore').strip('\x00')
                    section_names.append(name)
                    section_sizes.append(section.SizeOfRawData)
                    section_vsizes.append(section.VirtualSize)
                    if section.Characteristics & 0x20000000:
                        executable_sections_count += 1
                        code_section_size += section.SizeOfRawData
                        code_section_vsize += section.VirtualSize
                        if name.lower() not in common_executable_section_names:
                            pass
                    if section.Characteristics & 0x80000000:
                        writable_sections_count += 1
                    if section.Characteristics & 0x40000000:
                        readable_sections_count += 1
                        data_section_size += section.SizeOfRawData
                        data_section_vsize += section.VirtualSize
                    if (section.Characteristics & 0x20000000) and (section.Characteristics & 0x80000000):
                        rwx_sections_count += 1
                except Exception:
                    pass
            features['section_names_count'] = len(section_names)
            features['section_total_size'] = sum(section_sizes)
            features['section_total_vsize'] = sum(section_vsizes)
            features['avg_section_size'] = np.mean(section_sizes) if section_sizes else 0
            features['avg_section_vsize'] = np.mean(section_vsizes) if section_vsizes else 0
            features['max_section_size'] = np.max(section_sizes) if section_sizes else 0
            features['min_section_size'] = np.min(section_sizes) if section_sizes else 0
            features['code_section_ratio'] = code_section_size / (features['section_total_size'] + 1)
            features['data_section_ratio'] = data_section_size / (features['section_total_size'] + 1)
            features['code_vsize_ratio'] = code_section_vsize / (features['section_total_vsize'] + 1)
            features['data_vsize_ratio'] = data_section_vsize / (features['section_total_vsize'] + 1)
            features['executable_sections_count'] = executable_sections_count
            features['writable_sections_count'] = writable_sections_count
            features['readable_sections_count'] = readable_sections_count
            features['executable_sections_ratio'] = executable_sections_count / (len(section_names) + 1)
            features['writable_sections_ratio'] = writable_sections_count / (len(section_names) + 1)
            features['readable_sections_ratio'] = readable_sections_count / (len(section_names) + 1)
            features['rwx_sections_count'] = rwx_sections_count
            features['rwx_sections_ratio'] = rwx_sections_count / (len(section_names) + 1)
            if section_sizes:
                features['section_size_std'] = np.std(section_sizes)
                features['section_size_cv'] = np.std(section_sizes) / (np.mean(section_sizes) + 1e-8)
            else:
                features['section_size_std'] = 0
                features['section_size_cv'] = 0
            if section_names:
                section_name_lengths = [len(name) for name in section_names]
                features['section_name_avg_length'] = np.mean(section_name_lengths)
                features['section_name_max_length'] = np.max(section_name_lengths)
                features['section_name_min_length'] = np.min(section_name_lengths)
                lower_names = [n.lower() for n in section_names]
                features['has_upx_section'] = 1 if any('upx' in n for n in lower_names) else 0
                features['has_mpress_section'] = 1 if any('mpress' in n for n in lower_names) else 0
                features['has_aspack_section'] = 1 if any('aspack' in n for n in lower_names) else 0
                features['has_themida_section'] = 1 if any('themida' in n for n in lower_names) else 0
            else:
                features['section_name_avg_length'] = 0
                features['section_name_max_length'] = 0
                features['section_name_min_length'] = 0
                features['has_upx_section'] = 0
                features['has_mpress_section'] = 0
                features['has_aspack_section'] = 0
                features['has_themida_section'] = 0
        else:
            features['section_name_avg_length'] = 0
            features['section_name_max_length'] = 0
            features['section_name_min_length'] = 0
            features['max_section_size'] = 0
            features['min_section_size'] = 0
            features['code_section_ratio'] = 0
            features['data_section_ratio'] = 0
            features['code_vsize_ratio'] = 0
            features['data_vsize_ratio'] = 0
            features['section_size_std'] = 0
            features['section_size_cv'] = 0
            features['executable_sections_count'] = 0
            features['writable_sections_count'] = 0
            features['readable_sections_count'] = 0
            features['executable_sections_ratio'] = 0
            features['writable_sections_ratio'] = 0
            features['readable_sections_ratio'] = 0
            features['rwx_sections_count'] = 0
            features['rwx_sections_ratio'] = 0.0
        if hasattr(pe.OPTIONAL_HEADER, 'Subsystem'):
            features['subsystem'] = pe.OPTIONAL_HEADER.Subsystem
        else:
            features['subsystem'] = 0
        if hasattr(pe.OPTIONAL_HEADER, 'DllCharacteristics'):
            features['dll_characteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
            features['has_nx_compat'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x100 else 0
            features['has_aslr'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x40 else 0
            features['has_seh'] = 1 if not (pe.OPTIONAL_HEADER.DllCharacteristics & 0x400) else 0
            features['has_guard_cf'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x4000 else 0
        else:
            features['dll_characteristics'] = 0
            features['has_nx_compat'] = 0
            features['has_aslr'] = 0
            features['has_seh'] = 0
            features['has_guard_cf'] = 0
        features['has_resources'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') else 0
        features['has_debug_info'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') else 0
        features['has_tls'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0
        features['has_relocs'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0
        features['has_exceptions'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION') else 0
        try:
            tds = getattr(pe.FILE_HEADER, 'TimeDateStamp', 0)
            features['timestamp'] = int(tds) if tds else 0
            from datetime import datetime
            features['timestamp_year'] = datetime.utcfromtimestamp(int(tds)).year if tds else 0
        except Exception:
            features['timestamp'] = 0
            features['timestamp_year'] = 0
        try:
            sec_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY']]
            sig_size = getattr(sec_dir, 'Size', 0)
            features['has_signature'] = 1 if sig_size and sig_size > 0 else 0
            features['signature_size'] = sig_size if sig_size else 0
            try:
                va = getattr(sec_dir, 'VirtualAddress', 0)
                sz = getattr(sec_dir, 'Size', 0)
                if va and sz:
                    with open(valid_path, 'rb') as f:
                        f.seek(va)
                        blob = f.read(sz)
                    has_st = (b'signingTime' in blob) or (b'1.2.840.113549.1.9.5' in blob)
                    features['signature_has_signing_time'] = 1 if has_st else 0
                else:
                    features['signature_has_signing_time'] = 0
            except Exception:
                features['signature_has_signing_time'] = 0
        except Exception:
            features['has_signature'] = 0
            features['signature_size'] = 0
            features['signature_has_signing_time'] = 0
        version_info_present = 0
        company_name_len = 0
        product_name_len = 0
        file_version_len = 0
        original_filename_len = 0
        try:
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']])
            if hasattr(pe, 'FileInfo'):
                for fi in pe.FileInfo:
                    if hasattr(fi, 'StringTable'):
                        for st in fi.StringTable:
                            if hasattr(st, 'entries'):
                                version_info_present = 1
                                for k, v in st.entries.items():
                                    key = k.strip().lower()
                                    val = v.strip() if isinstance(v, str) else ''
                                    if key == 'companyname':
                                        company_name_len = max(company_name_len, len(val))
                                    elif key == 'productname':
                                        product_name_len = max(product_name_len, len(val))
                                    elif key == 'fileversion':
                                        file_version_len = max(file_version_len, len(val))
                                    elif key == 'originalfilename':
                                        original_filename_len = max(original_filename_len, len(val))
        except Exception:
            pass
        features['version_info_present'] = version_info_present
        features['company_name_len'] = company_name_len
        features['product_name_len'] = product_name_len
        features['file_version_len'] = file_version_len
        features['original_filename_len'] = original_filename_len
        try:
            pe_header_size = pe.OPTIONAL_HEADER.SizeOfHeaders
            features['pe_header_size'] = pe_header_size
            features['header_size_ratio'] = pe_header_size / (file_size + 1)
        except Exception:
            features['pe_header_size'] = 0
            features['header_size_ratio'] = 0
    except Exception:
        default_keys = [
            'sections_count','symbols_count','imports_count','exports_count','unique_imports','unique_dlls','unique_apis',
            'section_names_count','section_total_size','section_total_vsize','avg_section_size','avg_section_vsize',
            'subsystem','dll_characteristics','code_section_ratio','data_section_ratio','code_vsize_ratio','data_vsize_ratio',
            'has_nx_compat','has_aslr','has_seh','has_guard_cf','has_resources','has_debug_info','has_tls','has_relocs',
            'has_exceptions','dll_name_avg_length','dll_name_max_length','dll_name_min_length','section_name_avg_length',
            'section_name_max_length','section_name_min_length','export_name_avg_length','export_name_max_length',
            'export_name_min_length','max_section_size','min_section_size','long_sections_count','short_sections_count',
            'section_size_std','section_size_cv','executable_writable_sections','file_entropy_avg','file_entropy_min',
            'file_entropy_max','file_entropy_range','zero_byte_ratio','printable_byte_ratio','trailing_data_size',
            'trailing_data_ratio','imported_system_dlls_count','exports_density','has_large_trailing_data','pe_header_size',
            'header_size_ratio','file_entropy_std','file_entropy_q25','file_entropy_q75','file_entropy_median',
            'high_entropy_ratio','low_entropy_ratio','entropy_change_rate','entropy_change_std','executable_sections_count',
            'writable_sections_count','readable_sections_count','executable_sections_ratio','writable_sections_ratio',
            'readable_sections_ratio','executable_code_density','non_standard_executable_sections_count','rwx_sections_count',
            'rwx_sections_ratio','special_char_ratio','long_sections_ratio','short_sections_ratio'
        ]
        for key in default_keys:
            features[key] = 0
    return features

def extract_lightweight_pe_features(file_path):
    feature_vector = np.zeros(256, dtype=np.float32)
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            return feature_vector
        pe = pefile.PE(valid_path, fast_load=True)
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                if entry.dll:
                    dll_name = entry.dll.decode('utf-8').lower()
                    dll_hash = int(hashlib.sha256(dll_name.encode('utf-8')).hexdigest(), 16)
                    feature_vector[dll_hash % 128] = 1
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        api_name = imp.name.decode('utf-8')
                        api_hash = int(hashlib.sha256(api_name.encode('utf-8')).hexdigest(), 16)
                        feature_vector[128 + (api_hash % 128)] = 1
        if hasattr(pe, 'sections'):
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
                section_hash = int(hashlib.sha256(section_name.encode('utf-8')).hexdigest(), 16)
                feature_vector[(section_hash % 32) + 224] = 1
        norm = np.linalg.norm(feature_vector)
        if norm > 0 and not np.isnan(norm):
            feature_vector /= norm
        return feature_vector
    except Exception:
        return feature_vector

def extract_combined_pe_features(file_path):
    lightweight_features = extract_lightweight_pe_features(file_path)
    enhanced_features = extract_enhanced_pe_features(file_path)
    file_attrs = extract_file_attributes(file_path)
    all_features = {}
    all_features.update(enhanced_features)
    all_features.update(file_attrs)
    combined_vector = np.zeros(1000, dtype=np.float32)
    combined_vector[:256] = lightweight_features * 1.5
    max_file_size = 100 * 1024 * 1024
    max_timestamp = 2147483647
    feature_order = [
        'size','log_size','sections_count','symbols_count','imports_count','exports_count',
        'unique_imports','unique_dlls','unique_apis','section_names_count','section_total_size',
        'section_total_vsize','avg_section_size','avg_section_vsize','subsystem','dll_characteristics',
        'code_section_ratio','data_section_ratio','code_vsize_ratio','data_vsize_ratio',
        'has_nx_compat','has_aslr','has_seh','has_guard_cf','has_resources','has_debug_info',
        'has_tls','has_relocs','has_exceptions','dll_name_avg_length','dll_name_max_length',
        'dll_name_min_length','section_name_avg_length','section_name_max_length','section_name_min_length',
        'export_name_avg_length','export_name_max_length','export_name_min_length','max_section_size',
        'min_section_size','long_sections_count','short_sections_count','section_size_std','section_size_cv',
        'executable_writable_sections','file_entropy_avg','file_entropy_min','file_entropy_max','file_entropy_range',
        'zero_byte_ratio','printable_byte_ratio','trailing_data_size','trailing_data_ratio','imported_system_dlls_count',
        'exports_density','has_large_trailing_data','pe_header_size','header_size_ratio','file_entropy_std',
        'file_entropy_q25','file_entropy_q75','file_entropy_median','high_entropy_ratio','low_entropy_ratio',
        'entropy_change_rate','entropy_change_std','executable_sections_count','writable_sections_count',
        'readable_sections_count','executable_sections_ratio','writable_sections_ratio','readable_sections_ratio',
        'executable_code_density','non_standard_executable_sections_count','rwx_sections_count','rwx_sections_ratio',
        'special_char_ratio','long_sections_ratio','short_sections_ratio'
    ]
    common_sections = ['.text','.data','.rdata','.reloc','.rsrc']
    for sec in common_sections:
        feature_order.append(f'has_{sec}_section')
    feature_order.extend([
        'has_signature','signature_size','signature_has_signing_time','version_info_present','company_name_len','product_name_len','file_version_len','original_filename_len',
        'has_upx_section','has_mpress_section','has_aspack_section','has_themida_section','timestamp','timestamp_year'
    ])
    for i, key in enumerate(feature_order):
        if 256 + i >= 1000:
            break
        if key in all_features:
            val = all_features[key]
            if 'size' in key and isinstance(val, (int, float)):
                val = val / max_file_size
            elif key == 'timestamp' and isinstance(val, (int, float)):
                val = val / max_timestamp
            elif key == 'timestamp_year' and isinstance(val, (int, float)):
                val = (val - 1970) / (2038 - 1970)
            elif key.startswith('has_') and isinstance(val, (int, float)):
                val = float(val)
            elif key == 'log_size' and isinstance(val, (int, float)):
                val = val / np.log(max_file_size)
            combined_vector[256 + i] = val * 0.8 if isinstance(val, (int, float)) else 0
    norm = np.linalg.norm(combined_vector)
    if norm > 0 and not np.isnan(norm):
        combined_vector /= norm
    else:
        combined_vector = np.zeros(1000, dtype=np.float32)
    return combined_vector

def extract_features_in_memory(input_file_path, max_file_size=DEFAULT_MAX_FILE_SIZE):
    byte_sequence, orig_len = extract_byte_sequence(input_file_path, max_file_size)
    if byte_sequence is None:
        return None, None, 0
    pe_features = extract_combined_pe_features(input_file_path)
    return byte_sequence, pe_features, orig_len
