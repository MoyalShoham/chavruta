"""Outbound network allowlist.

The backend may talk to Sefaria, Hebcal and the Anthropic API, and to nothing else. This is
enforced twice:

1. Every HTTP client in the code is built by `http_client()`, whose transport refuses any other host
   (redirects included, since each hop goes through the transport).
2. `install_socket_guard()` wraps name resolution and every socket call that reaches an address
   (connect, connect_ex, sendto, sendmsg) for the whole process, so a library that opens its own
   connections is stopped too. Only allowlisted names (and the database host, which the caller
   passes in) resolve; an IP address is reachable only if it is loopback or an allowlisted name
   resolved to it. An IP literal never allowlists itself.
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
    """Process-wide guard. Idempotent. `extra_hosts` is for the database host.

    - Name resolution (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`) succeeds only for
      allowlisted names. Their addresses are remembered; an IP literal is never added.
    - Every way Python sockets reach an address (`connect`, `connect_ex`, `sendto`, `sendmsg`)
      requires a loopback address or one that an allowlisted name resolved to.

    Native code that opens its own sockets (libpq, for the configured database) is outside it.
    """
    global _installed
    extra = {h.lower() for h in extra_hosts}
    with _guard_lock:
        if _installed:
            return
        original_getaddrinfo = socket.getaddrinfo
        original_gethostbyname = socket.gethostbyname
        original_gethostbyname_ex = socket.gethostbyname_ex
        cls: Any = socket.socket
        original_connect = cls.connect
        original_connect_ex = cls.connect_ex
        original_sendto = cls.sendto
        original_sendmsg = cls.sendmsg

        def check_name(host: Any) -> str:
            name = host.decode() if isinstance(host, bytes) else str(host)
            if _is_ip(name):
                if not _ip_allowed(name):
                    raise BlockedHostError(f"connection to {name} is not allowed")
            elif not is_allowed(name, extra):
                raise BlockedHostError(f"name resolution for {name} is not allowed")
            return name

        def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if host is None:  # a local bind, not an outbound name
                return original_getaddrinfo(host, *args, **kwargs)
            name = check_name(host)
            results = original_getaddrinfo(host, *args, **kwargs)
            if not _is_ip(name):
                for *_, sockaddr in results:
                    _allowed_ips.add(str(sockaddr[0]))
            return results

        def guarded_gethostbyname(host: str) -> str:
            name = check_name(host)
            ip: str = original_gethostbyname(host)
            if not _is_ip(name):
                _allowed_ips.add(ip)
            return ip

        def guarded_gethostbyname_ex(host: str) -> Any:
            name = check_name(host)
            result = original_gethostbyname_ex(host)
            if not _is_ip(name):
                _allowed_ips.update(result[2])
            return result

        def check_address(sock: socket.socket, address: Any) -> None:
            if sock.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
                ip = str(address[0])
                if not _is_ip(ip):
                    check_name(ip)  # a name passed straight to connect: resolve-time rules
                elif not _ip_allowed(ip):
                    raise BlockedHostError(f"connection to {ip} is not allowed")

        def guarded_connect(self: socket.socket, address: Any) -> Any:
            check_address(self, address)
            return original_connect(self, address)

        def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
            check_address(self, address)
            return original_connect_ex(self, address)

        def guarded_sendto(self: socket.socket, data: Any, *args: Any) -> Any:
            check_address(self, args[-1])
            return original_sendto(self, data, *args)

        def guarded_sendmsg(self: socket.socket, buffers: Any, *args: Any) -> Any:
            if len(args) >= 3:
                check_address(self, args[2])
            return original_sendmsg(self, buffers, *args)

        socket.getaddrinfo = guarded_getaddrinfo
        socket.gethostbyname = guarded_gethostbyname
        socket.gethostbyname_ex = guarded_gethostbyname_ex
        cls.connect = guarded_connect
        cls.connect_ex = guarded_connect_ex
        cls.sendto = guarded_sendto
        cls.sendmsg = guarded_sendmsg
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
