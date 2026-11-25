"""User-scoped Windows DPAPI storage; never persist plaintext API credentials."""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path


class Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def credential_path() -> Path:
    if os.name != "nt":
        raise RuntimeError("Use DEEPSEEK_API_KEY on non-Windows systems")
    return Path(os.environ["LOCALAPPDATA"]) / "ContextBundles" / "credentials" / "deepseek.dpapi"


def _transform(data: bytes, protect: bool) -> bytes:
    if os.name != "nt":
        raise RuntimeError("DPAPI requires Windows")
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    destination = Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt.CryptProtectData if protect else crypt.CryptUnprotectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(destination)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(destination.data, destination.size)
    finally:
        kernel.LocalFree(destination.data)


def save_deepseek(key: str) -> Path:
    if not key.strip() or "\n" in key.strip():
        raise ValueError("Invalid credential")
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_transform(key.strip().encode(), True))
    return path


def load_deepseek() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MODEL_API_KEY")
    if key:
        return key
    path = credential_path()
    if not path.exists():
        raise RuntimeError("Configure DEEPSEEK_API_KEY or store a local DPAPI credential")
    return _transform(path.read_bytes(), False).decode()
