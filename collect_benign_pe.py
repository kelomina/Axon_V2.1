import os
import shutil
import hashlib
import pefile
from typing import Optional

from config.config import BENIGN_WHITELIST_PENDING_DIR, COLLECT_SOURCE_ROOT
from utils.path_utils import validate_path


def is_pe_file(file_path: str) -> bool:
    try:
        valid_path = validate_path(file_path)
        if not valid_path:
            return False
        with open(valid_path, 'rb') as f:
            sig = f.read(2)
            if sig != b'MZ':
                return False
        pe = pefile.PE(valid_path, fast_load=True)
        pe.close()
        return True
    except pefile.PEFormatError:
        return False
    except Exception:
        return False


def compute_sha256(file_path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_pe_files(source_root: Optional[str] = None, dest_dir: Optional[str] = None) -> int:
    src_root = source_root or COLLECT_SOURCE_ROOT
    dst_dir = dest_dir or BENIGN_WHITELIST_PENDING_DIR
    os.makedirs(dst_dir, exist_ok=True)
    total_copied = 0
    for root, _, files in os.walk(src_root):
        for name in files:
            src_path = os.path.join(root, name)
            if not is_pe_file(src_path):
                continue
            digest = compute_sha256(src_path)
            if not digest:
                continue
            dst_path = os.path.join(dst_dir, digest)
            if os.path.exists(dst_path):
                continue
            try:
                shutil.copy2(src_path, dst_path)
                total_copied += 1
            except Exception:
                pass
    return total_copied


def main() -> None:
    copied = collect_pe_files()
    print(f"copied={copied}")


if __name__ == '__main__':
    main()

