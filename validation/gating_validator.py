import os
import json
import argparse
from pathlib import Path
from features.extractor_in_memory import extract_enhanced_pe_features, extract_file_attributes
from config.config import GATE_HIGH_ENTROPY_RATIO, GATE_PACKED_SECTIONS_RATIO, GATE_PACKER_RATIO

def collect_signals(file_path: str) -> dict:
    attrs = extract_file_attributes(file_path)
    pe = extract_enhanced_pe_features(file_path)
    signals = {
        'high_entropy_ratio': float(attrs.get('high_entropy_ratio', 0.0)),
        'packed_sections_ratio': float(pe.get('packed_sections_ratio', 0.0)),
        'overlay_high_entropy_flag': int(pe.get('overlay_high_entropy_flag', 0)),
        'packer_keyword_hits_ratio': float(pe.get('packer_keyword_hits_ratio', 0.0)),
        'has_upx_section': int(pe.get('has_upx_section', 0)),
        'has_mpress_section': int(pe.get('has_mpress_section', 0)),
        'has_aspack_section': int(pe.get('has_aspack_section', 0)),
        'has_themida_section': int(pe.get('has_themida_section', 0)),
    }
    return signals

def decide(signals: dict) -> str:
    score = 0
    if float(signals.get('high_entropy_ratio', 0.0)) >= float(GATE_HIGH_ENTROPY_RATIO):
        score += 1
    if float(signals.get('packed_sections_ratio', 0.0)) >= float(GATE_PACKED_SECTIONS_RATIO):
        score += 1
    if int(signals.get('overlay_high_entropy_flag', 0)) == 1:
        score += 1
    if float(signals.get('packer_keyword_hits_ratio', 0.0)) >= float(GATE_PACKER_RATIO):
        score += 1
    if int(signals.get('has_upx_section', 0)) == 1 or int(signals.get('has_mpress_section', 0)) == 1 or int(signals.get('has_aspack_section', 0)) == 1 or int(signals.get('has_themida_section', 0)) == 1:
        score += 1
    return 'packed' if score >= 2 else 'normal'

def evaluate_directory(directory_path: str, recursive: bool = False) -> dict:
    files = Path(directory_path).rglob('*') if recursive else Path(directory_path).glob('*')
    files = [f for f in files if f.is_file()]
    packed = 0
    normal = 0
    details = []
    for f in files:
        s = collect_signals(str(f))
        d = decide(s)
        if d == 'packed':
            packed += 1
        else:
            normal += 1
        if len(details) < 50:
            details.append({'file_path': str(f), 'decision': d, 'signals': s})
    return {'total': len(files), 'packed': packed, 'normal': normal, 'details_sample': details}

def main():
    parser = argparse.ArgumentParser(description='Gating validation')
    parser.add_argument('--dir-path', type=str)
    parser.add_argument('--file-path', type=str)
    parser.add_argument('--recursive', '-r', action='store_true')
    args = parser.parse_args()
    if args.file_path:
        s = collect_signals(args.file_path)
        d = decide(s)
        print(json.dumps({'file_path': args.file_path, 'decision': d, 'signals': s}, ensure_ascii=False, indent=2))
        return
    if args.dir_path:
        stats = evaluate_directory(args.dir_path, args.recursive)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    parser.print_help()

if __name__ == '__main__':
    main()
