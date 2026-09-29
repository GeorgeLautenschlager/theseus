from __future__ import annotations

import ctypes as c
from unittest.mock import Mock

import pytest

from theseus.surrogates.desktop_observer import DesktopState
from theseus.surrogates.windows import desktop
from theseus.surrogates.windows.desktop import WindowsDesktopSource, SessionInfo, SessionInfoEx


def test_wts_structure_layout():
    # SDK ABI sizes include the eight-byte alignment of LARGE_INTEGER / union.
    assert c.sizeof(SessionInfo) == 224
    assert c.sizeof(SessionInfoEx) == 232
    assert SessionInfoEx.data.offset == 8


def fake_api(monkeypatch, flags=1, idle_ms=1000):
    monkeypatch.setattr(desktop.sys, 'platform', 'win32')
    monkeypatch.setattr(c, 'set_last_error', lambda value: None, raising=False)
    monkeypatch.setattr(c, 'get_last_error', lambda: 0, raising=False)
    user, kernel, wts = Mock(), Mock(), Mock()
    info = SessionInfoEx()
    info.level = 1
    info.data.flags = flags

    def session(pid, out):
        out._obj.value = 3
        return 1

    def query(server, session_id, kind, buffer, size):
        assert session_id == 3 and kind == 25
        buffer._obj.value = c.addressof(info)
        size._obj.value = c.sizeof(info)
        return 1

    def last_input(out):
        out._obj.time = (500 - idle_ms) & 0xFFFFFFFF
        return 1

    def pid(hwnd, out):
        out._obj.value = 42
        return 1

    def title(hwnd, out, size):
        out.value = 'Document'
        return len(out.value)

    def image(handle, flags, out, size):
        out.value = r'C:\Apps\editor.exe'
        return 1

    kernel.ProcessIdToSessionId.side_effect = session
    kernel.GetCurrentProcessId.return_value = 42
    wts.WTSQuerySessionInformationW.side_effect = query
    user.GetLastInputInfo.side_effect = last_input
    kernel.GetTickCount.return_value = 500
    user.GetForegroundWindow.return_value = 2**40  # catches accidental handle narrowing in callers
    user.GetWindowThreadProcessId.side_effect = pid
    user.GetWindowTextW.side_effect = title
    kernel.OpenProcess.return_value = 100
    kernel.QueryFullProcessImageNameW.side_effect = image
    source = WindowsDesktopSource()
    source._api = user, kernel, wts
    return source, user, kernel, wts


def test_native_sample_and_resources(monkeypatch):
    source, user, kernel, wts = fake_api(monkeypatch)
    state = source.sample()
    assert state == DesktopState(2**40, 42, 'editor.exe', 'Document', 'active', False)
    kernel.CloseHandle.assert_called_once_with(100)
    wts.WTSFreeMemory.assert_called_once()


def test_locked_skips_window_and_wrap_safe_idle(monkeypatch):
    source, user, kernel, wts = fake_api(monkeypatch, flags=0, idle_ms=61000)
    assert source.sample() == DesktopState(activity='idle', locked=True)
    user.GetForegroundWindow.assert_not_called()
    wts.WTSFreeMemory.assert_called_once()


def test_native_failures_are_unknown_not_locked(monkeypatch):
    source, user, kernel, wts = fake_api(monkeypatch)
    wts.WTSQuerySessionInformationW.side_effect = None
    wts.WTSQuerySessionInformationW.return_value = 0
    user.GetLastInputInfo.side_effect = None
    user.GetLastInputInfo.return_value = 0
    kernel.OpenProcess.return_value = None
    state = source.sample()
    assert state.locked is None and state.activity == 'unknown'
    assert state.process_name is None and state.title == 'Document'
    kernel.CloseHandle.assert_not_called()
    wts.WTSFreeMemory.assert_not_called()


def test_switch_during_sample_discards_window(monkeypatch):
    source, user, kernel, wts = fake_api(monkeypatch)
    user.GetForegroundWindow.side_effect = [100, 200]
    assert source.sample() == DesktopState(activity='active', locked=False)


def test_non_windows_requires_no_native_api(monkeypatch):
    monkeypatch.setattr(desktop.sys, 'platform', 'linux')
    assert WindowsDesktopSource().sample() == DesktopState()


@pytest.mark.skipif(desktop.sys.platform != 'win32', reason='live Windows session required')
def test_live_windows_sample():
    state = WindowsDesktopSource().sample()
    assert state.activity in ('active', 'idle', 'unknown')
    assert state.locked in (True, False, None)
