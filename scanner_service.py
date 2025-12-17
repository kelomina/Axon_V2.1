import os
import shutil
import tempfile
import sys
import asyncio
from pathlib import Path
from threading import Lock
from typing import Optional, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from scanner import MalwareScanner
from config.config import MODEL_PATH, FAMILY_CLASSIFIER_PATH, SCAN_CACHE_PATH, DEFAULT_MAX_FILE_SIZE, DEFAULT_SERVE_PORT, ENV_LIGHTGBM_MODEL_PATH, ENV_FAMILY_CLASSIFIER_PATH, ENV_CACHE_PATH, ENV_MAX_FILE_SIZE, ENV_SERVICE_PORT, ENV_ALLOWED_SCAN_ROOT, SERVICE_CONCURRENCY_LIMIT, SERVICE_PRINT_MALICIOUS_PATHS


ALLOWED_SCAN_ROOT = os.getenv(ENV_ALLOWED_SCAN_ROOT)

def _validate_user_path(path: str) -> Optional[str]:
    if not path:
        return None
    normalized = os.path.normpath(path)
    if '\0' in normalized:
        return None
    abs_path = os.path.abspath(normalized)
    if ALLOWED_SCAN_ROOT:
        base = os.path.abspath(ALLOWED_SCAN_ROOT)
        if not abs_path.startswith(base + os.sep) and abs_path != base:
            return None
    if not os.path.exists(abs_path):
        return None
    return abs_path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


class FileScanRequest(BaseModel):
    file_path: str = Field(..., description='需要扫描的文件绝对路径')


app = FastAPI(title='KoloVirusDetector Scanner Service', version='1.0.0')

_scanner_lock = Lock()
_scanner_instance: Optional[MalwareScanner] = None
_scan_semaphore = asyncio.Semaphore(SERVICE_CONCURRENCY_LIMIT)


def _prefer_gz(path: str) -> str:
    gz = path + ('.gz' if not path.endswith('.gz') else '')
    return gz if os.path.exists(gz) and not os.path.exists(path) else path

def _build_scanner() -> MalwareScanner:
    lightgbm_model_path = _prefer_gz(os.getenv(ENV_LIGHTGBM_MODEL_PATH, MODEL_PATH))
    family_classifier_path = _prefer_gz(os.getenv(ENV_FAMILY_CLASSIFIER_PATH, FAMILY_CLASSIFIER_PATH))
    cache_file = os.getenv(ENV_CACHE_PATH, SCAN_CACHE_PATH)
    max_file_size = _env_int(ENV_MAX_FILE_SIZE, DEFAULT_MAX_FILE_SIZE)

    missing_paths: List[str] = [
        p for p in [lightgbm_model_path, family_classifier_path]
        if not os.path.exists(p)
    ]
    if missing_paths:
        raise RuntimeError(f'以下必需文件不存在: {missing_paths}')

    return MalwareScanner(
        lightgbm_model_path=lightgbm_model_path,
        family_classifier_path=family_classifier_path,
        max_file_size=max_file_size,
        cache_file=None,
        enable_cache=False,
        print_malicious_paths=SERVICE_PRINT_MALICIOUS_PATHS,
    )


def get_scanner() -> MalwareScanner:
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = _build_scanner()
    return _scanner_instance


@app.on_event('startup')
def _startup() -> None:
    get_scanner()


@app.get('/health')
def health() -> dict:
    scanner = get_scanner()
    return {
        'status': 'ok',
        'cache_size': len(scanner.scan_cache),
        'model_path': os.getenv(ENV_LIGHTGBM_MODEL_PATH, MODEL_PATH)
    }


@app.post('/scan/file')
async def scan_file(request: FileScanRequest) -> dict:
    scanner = get_scanner()
    valid_path = _validate_user_path(request.file_path)
    if not valid_path:
        raise HTTPException(status_code=400, detail='路径不合法或不在允许的扫描目录内')

    async with _scan_semaphore:
        result = await asyncio.to_thread(scanner.scan_file, valid_path)

    if result is None:
        raise HTTPException(status_code=400, detail='文件不是有效的PE或扫描失败')
    result['virus_family'] = (result.get('malware_family') or {}).get('family_name')
    return result


@app.post('/scan/upload')
async def scan_upload(file: UploadFile = File(...)) -> dict:
    scanner = get_scanner()
    suffix = Path(file.filename or '').suffix or '.bin'

    def _scan_sync():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_path = tmp_file.name
        finally:
            file.file.close()
        try:
            result_local = scanner.scan_file(temp_path)
        finally:
            os.unlink(temp_path)
        return result_local

    async with _scan_semaphore:
        result = await asyncio.to_thread(_scan_sync)

    if result is None:
        raise HTTPException(status_code=400, detail='文件不是有效的PE或扫描失败')

    result['virus_family'] = (result.get('malware_family') or {}).get('family_name')
    result['original_filename'] = file.filename
    return result


@app.post('/cache/save')
async def flush_cache() -> dict:
    scanner = get_scanner()

    if not getattr(scanner, 'enable_cache', False):
        return {'status': 'disabled', 'cache_size': 0}

    async with _scan_semaphore:
        await asyncio.to_thread(scanner._save_cache)

    return {'status': 'saved', 'cache_size': len(scanner.scan_cache)}


if __name__ == '__main__':
    import uvicorn

    port = _env_int(ENV_SERVICE_PORT, DEFAULT_SERVE_PORT)
    uvicorn.run(app, host='0.0.0.0', port=port, reload=False)

