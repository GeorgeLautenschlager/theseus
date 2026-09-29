from __future__ import annotations

import ctypes as c
import math
import ntpath
import sys

from theseus.surrogates.desktop_observer import DesktopState

# Fixed-width Windows types keep structures inspectable on non-Windows hosts.
DWORD = c.c_uint32
LONG = c.c_int32
HANDLE = c.c_void_p
WCHAR = c.c_uint16


class LastInput(c.Structure):
    _fields_ = [("size", DWORD), ("time", DWORD)]


class SessionInfo(c.Structure):
    _fields_ = [
        ("session_id", DWORD), ("session_state", LONG), ("flags", LONG),
        ("station", WCHAR * 33), ("user", WCHAR * 21), ("domain", WCHAR * 18),
        ("logon", c.c_int64), ("connect", c.c_int64), ("disconnect", c.c_int64),
        ("last_input", c.c_int64), ("current", c.c_int64),
        ("incoming_bytes", DWORD), ("outgoing_bytes", DWORD),
        ("incoming_frames", DWORD), ("outgoing_frames", DWORD),
        ("incoming_compressed", DWORD), ("outgoing_compressed", DWORD),
    ]


class SessionInfoEx(c.Structure):
    _fields_ = [("level", DWORD), ("data", SessionInfo)]


class WindowsDesktopSource:
    """Current-session metadata, without screenshots, input hooks, or model calls.

    WTS session flags are interpreted for Windows 10/11. All native resources are
    released within a sample. DLLs are loaded lazily; construction is Linux-safe.
    """

    def __init__(self, idle_seconds: float = 60.0) -> None:
        if not math.isfinite(idle_seconds) or idle_seconds <= 0:
            raise ValueError("idle threshold must be finite and positive")
        self._idle_seconds = idle_seconds
        self._api = None

    def _load(self):
        user = c.WinDLL("user32", use_last_error=True)
        kernel = c.WinDLL("kernel32", use_last_error=True)
        wts = c.WinDLL("wtsapi32", use_last_error=True)
        signatures = [
            (user.GetForegroundWindow, [], HANDLE),
            (user.GetWindowThreadProcessId, [HANDLE, c.POINTER(DWORD)], DWORD),
            (user.GetWindowTextW, [HANDLE, c.c_wchar_p, c.c_int], c.c_int),
            (user.GetLastInputInfo, [c.POINTER(LastInput)], c.c_int),
            (kernel.GetTickCount, [], DWORD),
            (kernel.GetCurrentProcessId, [], DWORD),
            (kernel.ProcessIdToSessionId, [DWORD, c.POINTER(DWORD)], c.c_int),
            (kernel.OpenProcess, [DWORD, c.c_int, DWORD], HANDLE),
            (kernel.QueryFullProcessImageNameW, [HANDLE, DWORD, c.c_wchar_p, c.POINTER(DWORD)], c.c_int),
            (kernel.CloseHandle, [HANDLE], c.c_int),
            (wts.WTSQuerySessionInformationW, [HANDLE, DWORD, c.c_int, c.POINTER(HANDLE), c.POINTER(DWORD)], c.c_int),
            (wts.WTSFreeMemory, [HANDLE], None),
        ]
        for fn, args, result in signatures:
            fn.argtypes, fn.restype = args, result
        return user, kernel, wts

    def sample(self) -> DesktopState:
        if sys.platform != "win32":
            return DesktopState()
        if self._api is None:
            self._api = self._load()
        user, kernel, wts = self._api
        locked = self._locked(kernel, wts)
        last = LastInput(c.sizeof(LastInput), 0)
        activity = "unknown"
        if user.GetLastInputInfo(c.byref(last)):
            elapsed = ((kernel.GetTickCount() - last.time) & 0xFFFFFFFF) / 1000
            activity = "idle" if elapsed >= self._idle_seconds else "active"
        if locked is True:
            return DesktopState(activity=activity, locked=True)
        hwnd = user.GetForegroundWindow()
        if not hwnd:
            return DesktopState(activity=activity, locked=locked)
        pid = DWORD()
        user.GetWindowThreadProcessId(hwnd, c.byref(pid))
        title = c.create_unicode_buffer(1025)
        c.set_last_error(0)
        copied = user.GetWindowTextW(hwnd, title, len(title))
        title_value = title.value if copied or c.get_last_error() == 0 else None
        process_name = None
        process = kernel.OpenProcess(0x1000, False, pid.value) if pid.value else None
        if process:
            try:
                path = c.create_unicode_buffer(32768)
                size = DWORD(len(path))
                if kernel.QueryFullProcessImageNameW(process, 0, path, c.byref(size)):
                    process_name = ntpath.basename(path.value)
            finally:
                kernel.CloseHandle(process)
        # A switch during sampling must not combine two windows' metadata.
        if user.GetForegroundWindow() != hwnd:
            return DesktopState(activity=activity, locked=locked)
        return DesktopState(hwnd, pid.value or None, process_name, title_value, activity, locked)

    @staticmethod
    def _locked(kernel, wts) -> bool | None:
        session = DWORD()
        if not kernel.ProcessIdToSessionId(kernel.GetCurrentProcessId(), c.byref(session)):
            return None
        buffer, size = HANDLE(), DWORD()
        if not wts.WTSQuerySessionInformationW(None, session.value, 25, c.byref(buffer), c.byref(size)):
            return None
        try:
            if not buffer or size.value < c.sizeof(SessionInfoEx):
                return None
            info = c.cast(buffer, c.POINTER(SessionInfoEx)).contents
            if info.level != 1:
                return None
            return {0: True, 1: False}.get(info.data.flags)
        finally:
            wts.WTSFreeMemory(buffer)
