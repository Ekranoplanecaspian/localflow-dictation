"""Tie child processes to this process's lifetime (Windows job objects).

A child started with `subprocess.Popen` outlives a parent that is killed rather than closed:
`terminate()` does not reach it, and neither does a crash or Task Manager. That cost us real
memory twice - a killed engine left `llama-server` holding ~2.7 GB of VRAM, and a killed tray
app left a whole engine holding ~4 GB - so every long-lived child now joins a job object
created with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The kernel empties that job when the last
handle to it closes, which happens when this process ends however it ends.

The handle is deliberately never closed: it must live exactly as long as the process does.
"""

from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9

_JOB_HANDLE = None


def kill_on_close_job():
    """The process-wide job object, or None if the OS refuses to give us one."""
    global _JOB_HANDLE
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            return None
        _JOB_HANDLE = job
        return job
    except Exception as e:
        log.debug("job object unavailable: %s", e)
        return None


def assign(proc: subprocess.Popen, what: str = "child") -> None:
    """Put a child in the job so it cannot outlive us. Best effort: a machine policy can
    forbid nested jobs, and dictation must still work if it does."""
    import ctypes

    job = kill_on_close_job()
    if not job:
        return
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):  # type: ignore[attr-defined]
            log.debug("AssignProcessToJobObject(%s) failed: %s", what, ctypes.get_last_error())
    except Exception as e:
        log.debug("could not put %s in the job: %s", what, e)
