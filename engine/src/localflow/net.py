"""Downloads on real networks: the Windows proxy, a Hugging Face mirror, room on the disk.

**Proxy.** Python's HTTP clients (urllib, httpx under huggingface_hub) read a proxy typed into
Windows' settings, but not one set by an automatic configuration script (PAC) or found by
automatic detection (WPAD) - which is how most company networks hand theirs out - and the Rust
downloader behind Hugging Face's large files reads environment variables only. So the engine asks
WinHTTP, which does all of it, what proxy the download hosts go through, and exports the answer
as HTTPS_PROXY / HTTP_PROXY for everything in the process and every process it starts. Local
addresses always bypass it (the clean-up server is on 127.0.0.1). Proxy variables the user set
themselves win. Automatic detection can take seconds when there is nothing to find, so it runs
in the background; downloads wait for it (`wait_for_proxy`).

**Mirror.** Where huggingface.co is blocked, `network.hf_endpoint` (the Hub's Models page) points
downloads at a mirror. huggingface_hub reads HF_ENDPOINT once, when first imported, so it is set
before that and patched into the library when changed.

**Room.** A model download checks the free space first, and says how much it needs (`ensure_space`).
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import shutil
import sys
import threading
from pathlib import Path

log = logging.getLogger(__name__)

# Where models and the clean-up server come from, for the proxy question.
DOWNLOAD_URL = "https://huggingface.co/"
LOCAL = "localhost,127.0.0.1,::1"
SPACE_MARGIN = 512 * 2**20  # never fill the disk to the last byte

_proxy_resolved = threading.Event()
_proxy_started = False  # a lookup was started (by `serve`); nothing to wait for otherwise
proxy_in_use: str | None = None  # for the log and the self-check


class NotEnoughSpace(OSError):
    """A download would not fit. `needed` and `free` in bytes, `drive` like "C:"."""

    def __init__(self, what: str, needed: int, free: int, drive: str):
        self.what, self.needed, self.free, self.drive = what, needed, free, drive
        super().__init__(errno.ENOSPC, f"not enough disk space to download {what}: needs about {gb(needed)} "
                                       f"free on {drive}, and there is {gb(free)}")


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


# ---------------------------------------------------------------------------------------------
# proxy

def parse_proxy_list(value: str | None) -> str | None:
    """The proxy for HTTPS out of a WinHTTP/Internet Settings proxy string: "host:port",
    "http=h:p;https=h:p", or a list of several (the first is used). None for none."""
    if not value:
        return None
    entries = [e.strip() for e in value.replace(" ", ";").split(";") if e.strip()]
    by_scheme: dict[str, str] = {}
    plain: list[str] = []
    for e in entries:
        if "=" in e:
            scheme, _, addr = e.partition("=")
            by_scheme.setdefault(scheme.strip().lower(), addr.strip())
        else:
            plain.append(e)
    chosen = by_scheme.get("https") or by_scheme.get("http") or (plain[0] if plain else None)
    if not chosen:
        return None  # only socks= or ftp=: not something to hand to an HTTP client
    return chosen if "://" in chosen else f"http://{chosen}"


def parse_bypass(value: str | None) -> str:
    """Internet Settings' bypass list ("<local>;*.corp;10.*") as NO_PROXY, local addresses always."""
    hosts = [h.strip() for h in (value or "").replace(" ", ";").split(";") if h.strip()]
    # "*.corp" is ".corp" to HTTP clients: the domain and everything under it.
    hosts = [h[1:] if h.startswith("*.") else h for h in hosts if h.lower() != "<local>"]
    return ",".join(dict.fromkeys([*LOCAL.split(","), *hosts]))


def windows_proxy(url: str = DOWNLOAD_URL, pac_url: str | None = None) -> tuple[str | None, str]:
    """(proxy URL or None for a direct connection, bypass list) that Windows would use for
    `url`: a static proxy from Internet Settings, or the answer of the automatic configuration
    script or automatic detection. `pac_url` asks one script directly (for tests)."""
    if sys.platform != "win32":
        return None, LOCAL
    from ctypes import wintypes

    class IEConfig(ctypes.Structure):
        # Raw pointers, not LPWSTR: ctypes would turn those into Python strings, and the memory
        # behind them has to be freed with GlobalFree.
        _fields_ = [("fAutoDetect", wintypes.BOOL), ("lpszAutoConfigUrl", ctypes.c_void_p),
                    ("lpszProxy", ctypes.c_void_p), ("lpszProxyBypass", ctypes.c_void_p)]

    class AutoProxyOptions(ctypes.Structure):
        _fields_ = [("dwFlags", wintypes.DWORD), ("dwAutoDetectFlags", wintypes.DWORD),
                    ("lpszAutoConfigUrl", wintypes.LPCWSTR), ("lpvReserved", ctypes.c_void_p),
                    ("dwReserved", wintypes.DWORD), ("fAutoLogonIfChallenged", wintypes.BOOL)]

    class ProxyInfo(ctypes.Structure):
        _fields_ = [("dwAccessType", wintypes.DWORD), ("lpszProxy", ctypes.c_void_p),
                    ("lpszProxyBypass", ctypes.c_void_p)]

    winhttp = ctypes.WinDLL("winhttp", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    winhttp.WinHttpOpen.restype = ctypes.c_void_p
    winhttp.WinHttpOpen.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                    wintypes.DWORD]
    winhttp.WinHttpGetProxyForUrl.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR,
                                              ctypes.POINTER(AutoProxyOptions), ctypes.POINTER(ProxyInfo)]
    winhttp.WinHttpCloseHandle.argtypes = [ctypes.c_void_p]

    def take(p: int | None) -> str | None:
        """A string WinHTTP allocated: read, then freed."""
        if not p:
            return None
        s = ctypes.wstring_at(p)
        kernel32.GlobalFree(p)
        return s

    ie = IEConfig()
    static, bypass, auto_detect, auto_url = None, None, False, pac_url
    if pac_url is None and winhttp.WinHttpGetIEProxyConfigForCurrentUser(ctypes.byref(ie)):
        auto_detect = bool(ie.fAutoDetect)
        auto_url = take(ie.lpszAutoConfigUrl)
        static = take(ie.lpszProxy)
        bypass = take(ie.lpszProxyBypass)
    if auto_detect or auto_url:
        WINHTTP_ACCESS_TYPE_NO_PROXY = 1
        session = winhttp.WinHttpOpen("LocalFlow", WINHTTP_ACCESS_TYPE_NO_PROXY, None, None, 0)
        if session:
            try:
                opts = AutoProxyOptions()
                opts.dwFlags = (1 if auto_detect else 0) | (2 if auto_url else 0)  # AUTO_DETECT, CONFIG_URL
                opts.dwAutoDetectFlags = 3 if auto_detect else 0  # DHCP | DNS_A
                opts.lpszAutoConfigUrl = auto_url
                opts.fAutoLogonIfChallenged = True
                info = ProxyInfo()
                if winhttp.WinHttpGetProxyForUrl(session, url, ctypes.byref(opts), ctypes.byref(info)):
                    found, found_bypass = take(info.lpszProxy), take(info.lpszProxyBypass)
                    if info.dwAccessType == 3 and found:  # WINHTTP_ACCESS_TYPE_NAMED_PROXY
                        return parse_proxy_list(found), parse_bypass(found_bypass or bypass)
                    return None, parse_bypass(bypass)
                # No script found (the usual answer of automatic detection): fall through to the
                # static setting, as Windows itself does.
                log.debug("automatic proxy lookup found nothing (error %d)", ctypes.get_last_error())
            finally:
                winhttp.WinHttpCloseHandle(session)
    return parse_proxy_list(static), parse_bypass(bypass)


def apply_proxy(background: bool = True) -> None:
    """Export the Windows proxy to this process and its children. In the background by default:
    automatic detection can take seconds, and nothing at start-up waits for a download."""
    global _proxy_started
    _proxy_started = True
    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY"):
        log.info("using the proxy set in the environment")
        _proxy_resolved.set()
        return

    def resolve() -> None:
        global proxy_in_use
        try:
            proxy, bypass = windows_proxy()
            os.environ["NO_PROXY"] = os.environ.get("NO_PROXY") or bypass
            if proxy:
                os.environ["HTTPS_PROXY"] = os.environ["HTTP_PROXY"] = proxy
                proxy_in_use = proxy
                log.info("downloads go through the Windows proxy %s", proxy)
        except Exception as e:
            log.warning("could not read the Windows proxy settings: %s", e)
        finally:
            _proxy_resolved.set()

    if background:
        threading.Thread(target=resolve, name="proxy", daemon=True).start()
    else:
        resolve()


def wait_for_proxy(timeout: float = 15.0) -> None:
    """Before a download: let the proxy lookup finish, if it is still going."""
    if _proxy_started and not _proxy_resolved.wait(timeout):
        log.warning("the proxy lookup is taking long; downloading without waiting for it")


# ---------------------------------------------------------------------------------------------
# mirror

def apply_endpoint(endpoint: str) -> None:
    """Point Hugging Face downloads at `endpoint` ("" for huggingface.co), now and for every
    process started from here."""
    url = endpoint.strip().rstrip("/")
    if url:
        os.environ["HF_ENDPOINT"] = url
    else:
        os.environ.pop("HF_ENDPOINT", None)
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        # Already imported: it read HF_ENDPOINT then, so tell it again.
        new = url or "https://huggingface.co"
        constants.ENDPOINT = new
        constants.HUGGINGFACE_CO_URL_TEMPLATE = new + "/{repo_id}/resolve/{revision}/{filename}"
    if url:
        log.info("Hugging Face downloads come from %s", url)


def download_host() -> str:
    """The host models come from now: the mirror's, or huggingface.co."""
    from urllib.parse import urlparse

    return urlparse(os.environ.get("HF_ENDPOINT") or "https://huggingface.co").hostname or "huggingface.co"


# ---------------------------------------------------------------------------------------------
# disk

def ensure_space(folder: Path, needed: int, what: str) -> None:
    """Raise NotEnoughSpace unless `folder`'s drive has `needed` bytes free, and a margin."""
    folder = Path(folder)
    probe = folder
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed + SPACE_MARGIN:
        raise NotEnoughSpace(what, needed, free, probe.drive or str(probe.anchor))


def hf_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)
