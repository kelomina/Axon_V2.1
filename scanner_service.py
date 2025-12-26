import os
import signal
import asyncio
import json
import struct
from contextlib import suppress
from threading import Lock
from typing import Optional, List, Any, Dict, Tuple

from scanner import MalwareScanner
from config.config import (
    MODEL_PATH,
    FAMILY_CLASSIFIER_PATH,
    SCAN_CACHE_PATH,
    DEFAULT_MAX_FILE_SIZE,
    ENV_LIGHTGBM_MODEL_PATH,
    ENV_FAMILY_CLASSIFIER_PATH,
    ENV_CACHE_PATH,
    ENV_MAX_FILE_SIZE,
    ENV_ALLOWED_SCAN_ROOT,
    SERVICE_CONCURRENCY_LIMIT,
    SERVICE_PRINT_MALICIOUS_PATHS,
    SERVICE_EXIT_COMMAND,
    SERVICE_ADMIN_TOKEN,
    SERVICE_CONTROL_LOCALHOSTS,
    ENV_SERVICE_ADMIN_TOKEN,
    ENV_SERVICE_EXIT_COMMAND,
    SERVICE_MAX_BATCH_SIZE,
    SERVICE_IPC_HOST,
    SERVICE_IPC_PORT,
    SERVICE_IPC_MAX_MESSAGE_BYTES,
    SERVICE_IPC_READ_TIMEOUT_SEC,
    SERVICE_IPC_WRITE_TIMEOUT_SEC,
    SERVICE_IPC_REQUEST_TIMEOUT_SEC,
    SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION,
    ENV_SERVICE_IPC_HOST,
    ENV_SERVICE_IPC_PORT,
    ENV_SERVICE_IPC_MAX_MESSAGE_BYTES,
    ENV_SERVICE_IPC_READ_TIMEOUT_SEC,
    ENV_SERVICE_IPC_WRITE_TIMEOUT_SEC,
    ENV_SERVICE_IPC_REQUEST_TIMEOUT_SEC,
    ENV_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION,
)
from utils.logging_utils import get_logger


ALLOWED_SCAN_ROOT = os.getenv(ENV_ALLOWED_SCAN_ROOT)
_logger = get_logger('scanner_service')
_IPC_PROTOCOL_VERSION = 1
_ipc_server: Optional[asyncio.AbstractServer] = None
_ipc_server_task: Optional[asyncio.Task] = None

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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_ipc_host() -> str:
    return os.getenv(ENV_SERVICE_IPC_HOST, SERVICE_IPC_HOST)


def _get_ipc_port() -> int:
    return _env_int(ENV_SERVICE_IPC_PORT, SERVICE_IPC_PORT)


def _get_ipc_max_message_bytes() -> int:
    return _env_int(ENV_SERVICE_IPC_MAX_MESSAGE_BYTES, SERVICE_IPC_MAX_MESSAGE_BYTES)


def _get_ipc_read_timeout_sec() -> float:
    return _env_float(ENV_SERVICE_IPC_READ_TIMEOUT_SEC, float(SERVICE_IPC_READ_TIMEOUT_SEC))


def _get_ipc_write_timeout_sec() -> float:
    return _env_float(ENV_SERVICE_IPC_WRITE_TIMEOUT_SEC, float(SERVICE_IPC_WRITE_TIMEOUT_SEC))


def _get_ipc_request_timeout_sec() -> float:
    return _env_float(ENV_SERVICE_IPC_REQUEST_TIMEOUT_SEC, float(SERVICE_IPC_REQUEST_TIMEOUT_SEC))


def _get_ipc_max_requests_per_connection() -> int:
    return _env_int(ENV_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION, SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION)

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


def _get_exit_command() -> str:
    return os.getenv(ENV_SERVICE_EXIT_COMMAND, SERVICE_EXIT_COMMAND)


def _get_admin_token() -> str:
    return os.getenv(ENV_SERVICE_ADMIN_TOKEN, SERVICE_ADMIN_TOKEN)


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


def _ipc_response_ok(request_id: Optional[str], payload: Any) -> Dict[str, Any]:
    return {
        'version': _IPC_PROTOCOL_VERSION,
        'id': request_id,
        'ok': True,
        'payload': payload,
    }


def _ipc_response_error(request_id: Optional[str], code: str, message: str, details: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {'code': code, 'message': message}
    if details is not None:
        err['details'] = details
    return {
        'version': _IPC_PROTOCOL_VERSION,
        'id': request_id,
        'ok': False,
        'error': err,
    }


async def _ipc_read_message(reader: asyncio.StreamReader, max_bytes: int, timeout_sec: float) -> Optional[Dict[str, Any]]:
    header = await asyncio.wait_for(reader.readexactly(4), timeout=timeout_sec)
    size = struct.unpack('>I', header)[0]
    if size <= 0:
        raise ValueError('message_size_invalid')
    if size > max_bytes:
        raise ValueError('message_too_large')
    body = await asyncio.wait_for(reader.readexactly(size), timeout=timeout_sec)
    try:
        decoded = body.decode('utf-8')
        obj = json.loads(decoded)
    except Exception as e:
        raise ValueError('message_decode_failed') from e
    if not isinstance(obj, dict):
        raise ValueError('message_not_object')
    return obj


async def _ipc_write_message(writer: asyncio.StreamWriter, message: Dict[str, Any], max_bytes: int, timeout_sec: float) -> None:
    encoded = json.dumps(message, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > max_bytes:
        encoded = json.dumps(
            _ipc_response_error(message.get('id'), 'response_too_large', '响应体超过大小限制'),
            ensure_ascii=False,
            separators=(',', ':'),
        ).encode('utf-8')
    frame = struct.pack('>I', len(encoded)) + encoded
    writer.write(frame)
    await asyncio.wait_for(writer.drain(), timeout=timeout_sec)


def _ipc_extract_timeout_sec(msg: Dict[str, Any], default_timeout_sec: float) -> float:
    value = msg.get('timeout_ms')
    if value is None:
        return default_timeout_sec
    try:
        ms = float(value)
    except Exception:
        return default_timeout_sec
    if ms <= 0:
        return default_timeout_sec
    return ms / 1000.0


async def _ipc_handle_message(msg: Dict[str, Any], client_host: Optional[str]) -> Dict[str, Any]:
    request_id = msg.get('id')
    msg_type = msg.get('type')
    payload = msg.get('payload') or {}
    if not isinstance(payload, dict):
        return _ipc_response_error(request_id, 'invalid_payload', 'payload必须是对象')

    if msg.get('version', _IPC_PROTOCOL_VERSION) != _IPC_PROTOCOL_VERSION:
        return _ipc_response_error(request_id, 'version_mismatch', '协议版本不匹配')

    if msg_type == 'health':
        return _ipc_response_ok(request_id, {'status': 'ok'})

    if msg_type == 'scan_file':
        file_path = payload.get('file_path')
        if not isinstance(file_path, str):
            return _ipc_response_error(request_id, 'invalid_argument', 'file_path必须是字符串')
        valid_path = _validate_user_path(file_path)
        if not valid_path:
            return _ipc_response_error(request_id, 'invalid_path', '路径不合法或不在允许的扫描目录内')
        scanner = get_scanner()
        async with _scan_semaphore:
            result = await asyncio.to_thread(scanner.scan_file, valid_path)
        if result is None:
            return _ipc_response_error(request_id, 'scan_failed', '文件不是有效的PE或扫描失败')
        result['virus_family'] = (result.get('malware_family') or {}).get('family_name')
        return _ipc_response_ok(request_id, result)

    if msg_type == 'scan_batch':
        file_paths = payload.get('file_paths')
        if not isinstance(file_paths, list):
            return _ipc_response_error(request_id, 'invalid_argument', 'file_paths必须是数组')
        if len(file_paths) > SERVICE_MAX_BATCH_SIZE:
            return _ipc_response_error(request_id, 'batch_too_large', f'批量扫描数量超过限制: {SERVICE_MAX_BATCH_SIZE}')
        valid_paths: List[str] = []
        for p in file_paths:
            if not isinstance(p, str):
                continue
            vp = _validate_user_path(p)
            if vp:
                valid_paths.append(vp)
        if not valid_paths:
            return _ipc_response_ok(request_id, [])
        scanner = get_scanner()
        async with _scan_semaphore:
            results = await asyncio.to_thread(scanner.scan_batch, valid_paths)
        for res in results:
            res['virus_family'] = (res.get('malware_family') or {}).get('family_name')
        return _ipc_response_ok(request_id, results)

    if msg_type == 'control':
        command = payload.get('command')
        token = payload.get('token')
        if command != _get_exit_command():
            return _ipc_response_error(request_id, 'unknown_command', '未知控制指令')
        expected = _get_admin_token()
        if expected:
            if token != expected:
                return _ipc_response_error(request_id, 'forbidden', '无权限执行控制指令')
        else:
            if client_host not in set(SERVICE_CONTROL_LOCALHOSTS):
                return _ipc_response_error(request_id, 'forbidden', '无权限执行控制指令')
        asyncio.create_task(asyncio.to_thread(_cleanup_environment))
        asyncio.create_task(asyncio.to_thread(_trigger_process_exit))
        return _ipc_response_ok(request_id, {'status': 'shutting_down'})

    return _ipc_response_error(request_id, 'unknown_type', '未知消息类型')


async def _ipc_handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info('peername')
    client_host = peer[0] if isinstance(peer, (tuple, list)) and peer else None
    max_bytes = _get_ipc_max_message_bytes()
    read_timeout_sec = _get_ipc_read_timeout_sec()
    write_timeout_sec = _get_ipc_write_timeout_sec()
    default_request_timeout_sec = _get_ipc_request_timeout_sec()
    max_requests = _get_ipc_max_requests_per_connection()

    _logger.info(f'IPC连接建立: {client_host}')
    handled = 0
    try:
        while handled < max_requests:
            try:
                msg = await _ipc_read_message(reader, max_bytes=max_bytes, timeout_sec=read_timeout_sec)
            except asyncio.IncompleteReadError:
                break
            except asyncio.TimeoutError:
                await _ipc_write_message(
                    writer,
                    _ipc_response_error(None, 'timeout', '读取请求超时'),
                    max_bytes=max_bytes,
                    timeout_sec=write_timeout_sec,
                )
                break
            except ValueError as e:
                code = str(e)
                await _ipc_write_message(
                    writer,
                    _ipc_response_error(None, code, '请求解析失败'),
                    max_bytes=max_bytes,
                    timeout_sec=write_timeout_sec,
                )
                break
            except Exception:
                await _ipc_write_message(
                    writer,
                    _ipc_response_error(None, 'internal_error', '读取请求失败'),
                    max_bytes=max_bytes,
                    timeout_sec=write_timeout_sec,
                )
                break

            handled += 1
            msg_type = msg.get('type')
            request_id = msg.get('id')
            request_timeout_sec = min(_ipc_extract_timeout_sec(msg, default_request_timeout_sec), default_request_timeout_sec)
            try:
                response = await asyncio.wait_for(_ipc_handle_message(msg, client_host), timeout=request_timeout_sec)
            except asyncio.TimeoutError:
                response = _ipc_response_error(request_id, 'timeout', '处理请求超时')
            except Exception as e:
                _logger.error(f'IPC处理异常 type={msg_type} id={request_id} err={e}')
                response = _ipc_response_error(request_id, 'internal_error', '处理请求失败')

            try:
                await _ipc_write_message(writer, response, max_bytes=max_bytes, timeout_sec=write_timeout_sec)
            except Exception:
                break
    finally:
        with suppress(Exception):
            writer.close()
        with suppress(Exception):
            await writer.wait_closed()
        _logger.info(f'IPC连接关闭: {client_host}')


async def start_ipc_server(host: Optional[str] = None, port: Optional[int] = None) -> Tuple[str, int]:
    global _ipc_server, _ipc_server_task
    if _ipc_server is not None:
        sock = _ipc_server.sockets[0] if _ipc_server.sockets else None
        if sock is None:
            return _get_ipc_host(), _get_ipc_port()
        addr = sock.getsockname()
        return addr[0], addr[1]

    bind_host = host if host is not None else _get_ipc_host()
    bind_port = port if port is not None else _get_ipc_port()
    server = await asyncio.start_server(_ipc_handle_client, host=bind_host, port=bind_port, start_serving=True)
    _ipc_server = server
    _ipc_server_task = asyncio.create_task(server.serve_forever())
    sock = server.sockets[0] if server.sockets else None
    addr = sock.getsockname() if sock else (bind_host, bind_port)
    _logger.info(f'IPC服务启动: {addr[0]}:{addr[1]}')
    return addr[0], addr[1]


async def stop_ipc_server() -> None:
    global _ipc_server, _ipc_server_task
    task = _ipc_server_task
    server = _ipc_server
    _ipc_server_task = None
    _ipc_server = None

    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    if server is not None:
        server.close()
        with suppress(Exception):
            await server.wait_closed()
        _logger.info('IPC服务已关闭')


async def run_ipc_forever(host: Optional[str] = None, port: Optional[int] = None) -> None:
    get_scanner()
    await start_ipc_server(host=host, port=port)
    try:
        await asyncio.Event().wait()
    finally:
        with suppress(Exception):
            await stop_ipc_server()
        _cleanup_environment()


if __name__ == '__main__':
    asyncio.run(run_ipc_forever())

