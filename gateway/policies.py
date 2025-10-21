"""Per-app policy enforcement."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Optional

from .config import AppDefinition


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    reason: Optional[str] = None


class PolicyEngine:
    """Evaluates model allow/deny lists for applications."""

    def is_allowed(self, app: AppDefinition, model: str) -> PolicyDecision:
        policy = app.policy
        for pattern in policy.deny:
            if fnmatch.fnmatch(model, pattern):
                return PolicyDecision(allowed=False, reason=f"model '{model}' denied by policy")
        if policy.allow:
            for pattern in policy.allow:
                if fnmatch.fnmatch(model, pattern):
                    return PolicyDecision(allowed=True)
            return PolicyDecision(allowed=False, reason=f"model '{model}' not allowed by policy")
        return PolicyDecision(allowed=True)


__all__ = ["PolicyDecision", "PolicyEngine"]
