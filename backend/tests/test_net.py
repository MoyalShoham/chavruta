"""The outbound allowlist: only Sefaria, Hebcal and the Anthropic API.

The socket-guard tests run in a subprocess, because the guard is process-wide by design and must not
leak into the rest of the test session.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import httpx2 as httpx
import pytest

from chavruta.net import ALLOWED_HOSTS, AllowlistTransport, BlockedHostError, http_client, is_allowed


def test_allowlist_is_exactly_the_three_services() -> None:
    assert {h.removeprefix("www.") for h in ALLOWED_HOSTS} == {
        "sefaria.org",
        "hebcal.com",
        "api.anthropic.com",
    }


@pytest.mark.parametrize(
    "host",
    [
        "example.com",
        "sefaria.org.evil.com",
        "evilsefaria.org",
        "api.anthropic.com.attacker.net",
        "huggingface.co",
        "169.254.169.254",
    ],
)
def test_http_client_refuses_other_hosts(host: str) -> None:
    with http_client() as client, pytest.raises(BlockedHostError):
        client.get(f"https://{host}/")


def test_redirect_to_another_host_is_refused() -> None:
    """Each redirect hop goes through the transport, so a 302 off the allowlist is stopped."""

    class Redirecting(AllowlistTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "www.sefaria.org":
                return httpx.Response(302, headers={"Location": "https://example.com/steal"}, request=request)
            return super().handle_request(request)

    with (
        httpx.Client(transport=Redirecting(), follow_redirects=True) as client,
        pytest.raises(BlockedHostError, match=r"example\.com"),
    ):
        client.get("https://www.sefaria.org/api/texts/Genesis.1")


def test_is_allowed_normalises_case_and_trailing_dot() -> None:
    assert is_allowed("WWW.Sefaria.org.")
    assert is_allowed("db", extra=["db"])
    assert not is_allowed("db")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=60, check=False
    )


def test_socket_guard_blocks_libraries_that_bypass_the_client() -> None:
    """urllib and raw sockets never see AllowlistTransport; the process-wide guard stops them."""
    result = _run(
        """
        import socket, urllib.request
        from chavruta.net import BlockedHostError, install_socket_guard
        install_socket_guard(extra_hosts=["db"])
        for attempt in (
            lambda: urllib.request.urlopen("http://example.com/", timeout=5),
            lambda: socket.create_connection(("example.com", 80), timeout=5),
            lambda: socket.getaddrinfo("huggingface.co", 443),
        ):
            try:
                attempt()
            except BlockedHostError as e:
                print("blocked:", e)
            except OSError as e:
                if isinstance(e.__cause__, BlockedHostError) or "not allowed" in str(e):
                    print("blocked:", e)
                else:
                    print("NOT BLOCKED:", type(e).__name__, e)
            else:
                print("NOT BLOCKED")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("blocked:") == 3, result.stdout
    assert "NOT BLOCKED" not in result.stdout


def test_socket_guard_blocks_raw_ip_literals_but_allows_loopback() -> None:
    result = _run(
        """
        import socket
        from chavruta.net import BlockedHostError, install_socket_guard
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        install_socket_guard()
        s = socket.socket()
        s.settimeout(5)
        try:
            s.connect(("93.184.215.14", 80))
        except BlockedHostError:
            print("public ip blocked")
        c = socket.socket()
        c.connect(server.getsockname())
        print("loopback ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "public ip blocked" in result.stdout
    assert "loopback ok" in result.stdout


def test_socket_guard_allows_the_three_services_and_the_database() -> None:
    """Resolution only (no traffic): the allowed names pass the guard, the database name too."""
    result = _run(
        """
        import socket
        from chavruta.net import BlockedHostError, install_socket_guard
        install_socket_guard(extra_hosts=["db.internal"])
        for host in ("www.sefaria.org", "www.hebcal.com", "api.anthropic.com", "db.internal"):
            try:
                socket.getaddrinfo(host, 443)
                print("resolved", host)
            except BlockedHostError:
                print("BLOCKED", host)
            except OSError:
                print("passed guard", host)  # offline CI or no such name: the guard let it through
        """
    )
    assert result.returncode == 0, result.stderr
    assert "BLOCKED" not in result.stdout, result.stdout


def test_ip_literals_cannot_bypass_the_guard_through_any_socket_path() -> None:
    """The review's bypass: getaddrinfo on an IP literal used to allowlist that IP, so
    create_connection, urlopen and a plain httpx client reached any address. 192.0.2.0/24 is
    TEST-NET-1, never routed, so nothing leaves the machine even if a check failed."""
    result = _run(
        """
        import socket, urllib.request
        import httpx2
        from chavruta.net import BlockedHostError, install_socket_guard
        install_socket_guard()
        target = ("192.0.2.1", 80)

        def udp():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(b"x", ("192.0.2.1", 53))

        def udp_msg():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendmsg([b"x"], [], 0, ("192.0.2.1", 53))

        attempts = {
            "getaddrinfo": lambda: socket.getaddrinfo("192.0.2.1", 80),
            "create_connection": lambda: socket.create_connection(target, timeout=2),
            "urlopen": lambda: urllib.request.urlopen("http://192.0.2.1/", timeout=2),
            "httpx": lambda: httpx2.get("http://192.0.2.1/", timeout=2),
            "connect": lambda: socket.socket().connect(target),
            "connect_ex": lambda: socket.socket().connect_ex(target),
            "connect_by_name": lambda: socket.socket().connect(("example.com", 80)),
            "sendto": udp,
            "sendmsg": udp_msg,
            "gethostbyname": lambda: socket.gethostbyname("example.com"),
            "gethostbyname_ex": lambda: socket.gethostbyname_ex("example.com"),
        }
        for name, attempt in attempts.items():
            try:
                attempt()
            except Exception as e:
                chain, blocked = e, False
                while chain is not None:
                    blocked = blocked or isinstance(chain, BlockedHostError) or "not allowed" in str(chain)
                    chain = chain.__cause__ or chain.__context__
                print(name, "blocked" if blocked else f"NOT BLOCKED ({type(e).__name__}: {e})")
            else:
                print(name, "NOT BLOCKED (succeeded)")
        """
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 11, result.stdout
    assert all(line.endswith(" blocked") for line in lines), result.stdout
