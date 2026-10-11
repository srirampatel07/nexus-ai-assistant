"""SSRF protection tests (Phase 6). No real DNS: resolvers are fakes."""

import pytest

from app.core.exceptions import ToolDenied
from app.web.errors import UnsafeDestinationError
from app.web.ssrf import check_url


def _resolve(mapping):
    def fake(host):
        if host in mapping:
            return [(2, 1, 6, "", (mapping[host], 0))]
        raise OSError(f"no such host: {host}")

    return fake


_PUBLIC = _resolve({"example.com": "93.184.216.34", "example.org": "93.184.216.34"})


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "http://example.com:8080/path?q=1",
        "https://example.org/a/b",
        "HTTPS://EXAMPLE.COM/UPPER",
    ],
)
def test_public_destinations_allowed(url):
    assert check_url(url, dns_resolver=_PUBLIC) == url.strip()


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://localhost:8000/x",
        "http://LOCALHOST/",
        "http://127.0.0.1/",
        "http://127.1.2.3/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://10.0.0.5/",
        "http://172.16.4.2/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/",
        "http://metadata.goog/",
        "http://example.local/",
        "http://app.internal/",
        "http://host.invalid/",
        "http://intranet/",
        "http://224.0.0.1/",
        "ftp://example.com/file",
        "file:///etc/passwd",
        "http://user:pass@example.com/",
        "gopher://example.com/",
        "",
        "   ",
        "not a url",
        "http:///no-host",
    ],
)
def test_restricted_destinations_refused(url):
    with pytest.raises(UnsafeDestinationError):
        check_url(url, dns_resolver=_PUBLIC)


@pytest.mark.parametrize(
    "host,ip",
    [
        ("evil.test", "10.1.2.3"),
        ("corp.test", "192.168.9.9"),
        ("loop.test", "127.0.0.1"),
        ("link.test", "169.254.20.10"),
        ("v6.test", "::1"),
        ("multi.test", "10.9.9.9"),
    ],
)
def test_dns_resolving_private_is_refused(host, ip):
    with pytest.raises(UnsafeDestinationError):
        check_url(f"https://{host}/", dns_resolver=_resolve({host: ip}))


def test_dns_addresses_all_checked_not_just_first():
    def mixed(host):
        return [
            (2, 1, 6, "", ("93.184.216.34", 0)),
            (2, 1, 6, "", ("10.0.0.9", 0)),
        ]

    with pytest.raises(UnsafeDestinationError):
        check_url("https://example.com/", dns_resolver=mixed)


def test_dns_failure_is_fail_closed():
    with pytest.raises(UnsafeDestinationError) as exc:
        check_url("https://ghost.nosuchdomain123/", dns_resolver=_resolve({}))
    assert "resolv" in str(exc.value).lower()


def test_refusal_is_a_policy_denial_without_secrets():
    with pytest.raises(ToolDenied):
        check_url("http://admin:s3cret@example.com/", dns_resolver=_PUBLIC)
    try:
        check_url("http://admin:s3cret@example.com/", dns_resolver=_PUBLIC)
    except UnsafeDestinationError as exc:
        assert "s3cret" not in str(exc)
        assert "admin" not in str(exc)


def test_public_ip_literals_allowed():
    assert check_url("http://93.184.216.34/", dns_resolver=_PUBLIC)
    assert check_url("http://8.8.8.8/", dns_resolver=_PUBLIC)
