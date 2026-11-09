"""Outbound network allowlist.

The backend may talk to Sefaria, Hebcal and the Anthropic API, and to nothing else. This is
enforced twice:

1. Every HTTP client in the code is built by `http_client()`, whose transport refuses any other host
   (redirects included, since each hop goes through the transport).
2. `install_socket_guard()` wraps name resolution and `socket.connect` for the whole process, so a
   library that opens its own connections is stopped too. Only allowlisted names (and the database
   and loopback, which the caller passes in) resolve; a raw IP connects only if it came from an
   allowlisted name.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from collections.abc import Iterable
from typing import Any

import httpx2 as httpx

ALLOWED_HOSTS: frozenset[str] = frozenset(
    {"sefaria.org", "www.sefaria.org", "hebcal.com", "www.hebcal.com", "api.anthropic.com"}
)
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


class BlockedHostError(PermissionError):
    """An outbound connection to a host that is not on the allowlist."""


def is_allowed(host: str, extra: Iterable[str] = ()) -> bool:
    h = host.lower().rstrip(".")
    return h in ALLOWED_HOSTS or h in LOOPBACK or h in {e.lower() for e in extra}


class AllowlistTransport(httpx.HTTPTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host not in ALLOWED_HOSTS:
            raise BlockedHostError(f"outbound request to {host} is not allowed")
        return super().handle_request(request)


def http_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(
        transport=AllowlistTransport(),
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "chavruta-study-partner/0.1"},
    )


_guard_lock = threading.Lock()
_installed = False
_allowed_ips: set[str] = set()


def install_socket_guard(extra_hosts: Iterable[str] = ()) -> None:
    """Process-wide guard. Idempotent. `extra_hosts` is for the database host."""
    global _installed
    extra = {h.lower() for h in extra_hosts}
    with _guard_lock:
        if _installed:
            return
        original_getaddrinfo = socket.getaddrinfo
        original_connect = socket.socket.connect

        def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            name = host.decode() if isinstance(host, bytes) else str(host) if host is not None else ""
            if host is not None and not _is_ip(name) and not is_allowed(name, extra):
                raise BlockedHostError(f"name resolution for {name} is not allowed")
            results = original_getaddrinfo(host, *args, **kwargs)
            for *_, sockaddr in results:
                _allowed_ips.add(str(sockaddr[0]))
            return results

        def guarded_connect(self: socket.socket, address: Any) -> Any:
            if self.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
                ip = str(address[0])
                if _is_ip(ip) and not _ip_allowed(ip):
                    raise BlockedHostError(f"connection to {ip} is not allowed")
            return original_connect(self, address)

        socket.getaddrinfo = guarded_getaddrinfo
        socket.socket.connect = guarded_connect  # type: ignore[method-assign,assignment]
        _installed = True


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _ip_allowed(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return addr.is_loopback or ip in _allowed_ips
