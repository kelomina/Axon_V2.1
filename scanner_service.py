import os
import shutil
import tempfile
import signal
import asyncio
import concurrent.futures
from pathlib import Path
from threading import Lock
from typing import Optional, List, Callable, Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
import numpy as np

from scanner import MalwareScanner
from config.config import MODEL_PATH, FAMILY_CLASSIFIER_PATH, SCAN_CACHE_PATH, DEFAULT_MAX_FILE_SIZE, DEFAULT_SERVE_PORT, ENV_LIGHTGBM_MODEL_PATH, ENV_FAMILY_CLASSIFIER_PATH, ENV_CACHE_PATH, ENV_MAX_FILE_SIZE, ENV_SERVICE_PORT, ENV_ALLOWED_SCAN_ROOT, SERVICE_CONCURRENCY_LIMIT, SERVICE_PRINT_MALICIOUS_PATHS, SERVICE_EXIT_COMMAND, SERVICE_ADMIN_TOKEN, SERVICE_CONTROL_LOCALHOSTS, ENV_SERVICE_ADMIN_TOKEN, ENV_SERVICE_EXIT_COMMAND, SERVICE_MAX_BATCH_SIZE, ENV_SERVICE_FEATURE_PROCESS_WORKERS, ENV_SERVICE_FEATURE_PROCESS_POOL_ENABLED, SERVICE_FEATURE_PROCESS_WORKERS, SERVICE_FEATURE_PROCESS_POOL_ENABLED, SERVICE_IO_THREAD_WORKERS


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


class FileBatchRequest(BaseModel):
    file_paths: List[str] = Field(..., description='需要扫描的文件路径列表')


class ControlRequest(BaseModel):
    command: str = Field(..., description='控制指令')
    token: Optional[str] = Field(None, description='管理令牌')


app = FastAPI(title='KoloVirusDetector Scanner Service', version='1.0.0')

_scanner_lock = Lock()
_scanner_instance: Optional[MalwareScanner] = None
_scan_semaphore = asyncio.Semaphore(SERVICE_CONCURRENCY_LIMIT)
_io_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_feature_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {'1', 'true', 'yes', 'y', 'on'}


def _extract_features_worker(file_path: str, max_file_size: int):
    try:
        import numpy as _np
        from config.config import PE_FEATURE_VECTOR_DIM
        from features.extractor_in_memory import extract_features_in_memory
        from features.statistics import extract_statistical_features

        byte_sequence, pe_features, orig_length = extract_features_in_memory(file_path, max_file_size)
        if byte_sequence is None or pe_features is None:
            return None

        orig_pe_len = len(pe_features)
        if orig_pe_len != PE_FEATURE_VECTOR_DIM:
            fixed_pe_features = _np.zeros(PE_FEATURE_VECTOR_DIM, dtype=_np.float32)
            copy_len = min(orig_pe_len, PE_FEATURE_VECTOR_DIM)
            fixed_pe_features[:copy_len] = pe_features[:copy_len]
            pe_features = fixed_pe_features

        features = extract_statistical_features(byte_sequence, pe_features, orig_length)
        return features
    except Exception:
        return None


async def _run_in_io(func: Callable[..., Any], *args: Any) -> Any:
    loop = asyncio.get_running_loop()
    if _io_executor is not None:
        return await loop.run_in_executor(_io_executor, func, *args)
    return await asyncio.to_thread(func, *args)


async def _extract_features_async(file_path: str) -> Optional[np.ndarray]:
    max_file_size = _env_int(ENV_MAX_FILE_SIZE, DEFAULT_MAX_FILE_SIZE)
    loop = asyncio.get_running_loop()
    if _feature_executor is not None:
        return await loop.run_in_executor(_feature_executor, _extract_features_worker, file_path, max_file_size)
    if _io_executor is not None:
        return await loop.run_in_executor(_io_executor, _extract_features_worker, file_path, max_file_size)
    return await asyncio.to_thread(_extract_features_worker, file_path, max_file_size)


def _build_result(scanner: MalwareScanner, file_path: str, features: np.ndarray, is_malware: bool, confidence: float) -> dict:
    result = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_size': os.path.getsize(file_path),
        'is_malware': bool(is_malware),
        'confidence': float(confidence),
    }

    if is_malware:
        cluster_id, family_name, is_new_family = scanner.predict_family(features)
        result['malware_family'] = {
            'cluster_id': int(cluster_id) if cluster_id is not None else -1,
            'family_name': family_name,
            'is_new_family': bool(is_new_family),
        }

    result['virus_family'] = (result.get('malware_family') or {}).get('family_name')
    return result


async def _scan_one_path(scanner: MalwareScanner, file_path: str) -> Optional[dict]:
    features = await _extract_features_async(file_path)
    if features is None:
        return None

    is_malware, confidence = scanner._predict_malware_from_features(features)
    return _build_result(scanner, file_path, features, is_malware, confidence)


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


def _get_exit_command() -> str:
    return os.getenv(ENV_SERVICE_EXIT_COMMAND, SERVICE_EXIT_COMMAND)


def _get_admin_token() -> str:
    return os.getenv(ENV_SERVICE_ADMIN_TOKEN, SERVICE_ADMIN_TOKEN)


def _is_authorized_control(request: Request, token: Optional[str]) -> bool:
    expected = _get_admin_token()
    if expected:
        return token == expected

    client = getattr(request, 'client', None)
    host = getattr(client, 'host', None) if client else None
    return host in set(SERVICE_CONTROL_LOCALHOSTS)


def _cleanup_environment() -> None:
    global _scanner_instance

    with _scanner_lock:
        scanner = _scanner_instance
        _scanner_instance = None

    if scanner is None:
        return

    try:
        if getattr(scanner, 'enable_cache', False):
            scanner._save_cache()
    except Exception:
        pass

    try:
        if hasattr(scanner, 'scan_cache'):
            scanner.scan_cache.clear()
    except Exception:
        pass

    try:
        temp_model_path = getattr(scanner, '_temp_model_path', None)
        if temp_model_path:
            os.unlink(temp_model_path)
    except Exception:
        pass


def _trigger_process_exit() -> None:
    os.kill(os.getpid(), signal.SIGINT)


@app.on_event('startup')
def _startup() -> None:
    get_scanner()
    global _io_executor, _feature_executor
    _io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=SERVICE_IO_THREAD_WORKERS)
    enabled = _env_bool(ENV_SERVICE_FEATURE_PROCESS_POOL_ENABLED, SERVICE_FEATURE_PROCESS_POOL_ENABLED)
    workers = _env_int(ENV_SERVICE_FEATURE_PROCESS_WORKERS, SERVICE_FEATURE_PROCESS_WORKERS)
    if enabled and workers > 0:
        _feature_executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)


@app.on_event('shutdown')
def _shutdown() -> None:
    global _io_executor, _feature_executor
    if _feature_executor is not None:
        try:
            _feature_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                _feature_executor.shutdown(wait=False)
            except Exception:
                pass
        _feature_executor = None
    if _io_executor is not None:
        try:
            _io_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                _io_executor.shutdown(wait=False)
            except Exception:
                pass
        _io_executor = None
    _cleanup_environment()


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
        result = await _scan_one_path(scanner, valid_path)

    if result is None:
        raise HTTPException(status_code=400, detail='文件不是有效的PE或扫描失败')
    return result


@app.post('/scan/batch')
async def scan_batch(request: FileBatchRequest) -> List[dict]:
    scanner = get_scanner()
    
    if len(request.file_paths) > SERVICE_MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f'批量扫描数量超过限制: {SERVICE_MAX_BATCH_SIZE}')
    
    # Pre-validate to save time on obviously wrong requests? 
    # Or just let scanner handle it. 
    # Let's validate here to ensure we only pass valid paths to scanner if possible, 
    # but scanner does validation too.
    # However, to respect ALLOWED_SCAN_ROOT from service env, we should check.
    
    valid_paths = []
    for p in request.file_paths:
        vp = _validate_user_path(p)
        if vp:
            valid_paths.append(vp)
            
    if not valid_paths:
        return []

    async with _scan_semaphore:
        tasks = []
        scan_paths = []
        for p in valid_paths:
            scan_paths.append(p)
            tasks.append(_extract_features_async(p))

        if not tasks:
            return []

        features_list = await asyncio.gather(*tasks)

        final_paths = []
        final_features = []
        for p, feat in zip(scan_paths, features_list):
            if feat is None:
                continue
            final_paths.append(p)
            final_features.append(feat)

        if not final_features:
            return []

        matrix = np.array(final_features)
        is_malware_list, confidence_list = scanner._predict_malware_batch(matrix)

        results: List[dict] = []
        for p, feat, is_malware, confidence in zip(final_paths, final_features, is_malware_list, confidence_list):
            res = _build_result(scanner, p, feat, bool(is_malware), float(confidence))
            results.append(res)

    return results


@app.post('/scan/upload')
async def scan_upload(file: UploadFile = File(...)) -> dict:
    scanner = get_scanner()
    suffix = Path(file.filename or '').suffix or '.bin'

    async with _scan_semaphore:
        try:
            def _save_temp() -> str:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    shutil.copyfileobj(file.file, tmp_file)
                    return tmp_file.name

            temp_path = await _run_in_io(_save_temp)
        finally:
            try:
                file.file.close()
            except Exception:
                pass

        try:
            result = await _scan_one_path(scanner, temp_path)
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass

    if result is None:
        raise HTTPException(status_code=400, detail='文件不是有效的PE或扫描失败')

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


@app.post('/control/command')
async def control_command(control: ControlRequest, request: Request, background_tasks: BackgroundTasks) -> dict:
    if control.command != _get_exit_command():
        raise HTTPException(status_code=400, detail='未知控制指令')

    if not _is_authorized_control(request, control.token):
        raise HTTPException(status_code=403, detail='无权限执行控制指令')

    background_tasks.add_task(_cleanup_environment)
    background_tasks.add_task(_trigger_process_exit)

    return {'status': 'shutting_down'}


if __name__ == '__main__':
    import uvicorn

    port = _env_int(ENV_SERVICE_PORT, DEFAULT_SERVE_PORT)
    uvicorn.run(app, host='0.0.0.0', port=port, reload=False)

