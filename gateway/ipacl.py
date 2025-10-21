"""CIDR-based ACL checks."""
from __future__ import annotations

import ipaddress
from typing import Iterable


def is_allowed(remote_ip: str, allowlist: Iterable[str]) -> bool:
    """Return True if the IP is allowed based on the given CIDR list."""
    try:
        ip_obj = ipaddress.ip_address(remote_ip)
    except ValueError:
        return False

    for cidr in allowlist:
        try:
            network = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            continue
        if ip_obj in network:
            return True
    return False


__all__ = ["is_allowed"]
