# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading the degradation record back inside a test.

Every place the run does something other than what was asked logs `degraded[KIND]: DETAIL` where it
happens; there is no ledger object to query. So the log IS the record, and a test that asserts on it
has to read the log — which pytest's `caplog` cannot do, because the engine logs through loguru and
`caplog` only sees `logging`. This fixture is the way in.

It is also per-test isolation, which the ledger needed a fixture for and this gets for free: the sink
is added and removed around one test, so nothing another test emitted can be seen here and nothing
emitted here survives into the next.
"""

from __future__ import annotations

import collections
import re

import pytest


class Degradations:
    """The `degraded[...]` lines one test provoked, parsed back into (kind, detail) pairs."""

    # DOTALL: several sites log a multi-line detail (an exception's own message, a joined list of
    # causes). Without it the row is silently truncated at the first newline and a `first()`
    # assertion on the tail of a detail fails for a reason that has nothing to do with the code.
    _ROW = re.compile(r"degraded\[([^\]]*)\]: (.*)", re.S)

    def __init__(self, lines: list[str]):
        self._lines = lines

    @property
    def rows(self) -> list[tuple[str, str]]:
        return [m.groups() for m in (self._ROW.search(x) for x in self._lines) if m]

    def kinds(self) -> collections.Counter:
        return collections.Counter(k for k, _ in self.rows)

    def first(self, kind: str) -> str:
        """The detail of the FIRST line of `kind`. Raises rather than returning '' when there is
        none: an assertion on a detail that was never logged has to fail as a missing record, not as
        a mismatched string."""
        for k, d in self.rows:
            if k == kind:
                return d
        raise KeyError(f"no degraded[{kind}] line; saw {dict(self.kinds())}")

    def __contains__(self, kind: str) -> bool:
        # EXACT match, and loud about the near miss. Five kinds are composite
        # (`api_gave_up:rate_limited`, `judge_sample_dropped:ValueError`, `op_failed:dilate`, …), so
        # `"api_gave_up" in degradations` is False however many of them fired — and a `not in`
        # assertion written that way passes forever, asserting nothing. The two idioms already sit
        # side by side in one test file, so the trap is half set; this springs it instead.
        seen = [k for k, _ in self.rows]
        if kind not in seen and any(k.startswith(kind + ":") for k in seen):
            raise AssertionError(
                f"{kind!r} is a PREFIX of a composite kind, not a kind: saw {sorted(set(seen))}. "
                f"Match the whole label, or test `any(k.startswith(...) for k in degradations.kinds())`."
            )
        return kind in seen

    def clear(self) -> None:
        self._lines.clear()

    def __repr__(self) -> str:
        return f"Degradations({self.rows!r})"


@pytest.fixture
def degradations():
    # Imported HERE, not at module scope, and that is load-bearing: `cosmos_framework.utils.log`
    # calls `init_loguru_stdout()` at import time, which calls a bare `logger.remove()` — it would
    # detach this fixture's sink if the import happened after `logger.add`. The symptom would be a
    # ValueError in teardown rather than a failed assertion about the code under test.
    from cosmos_framework.utils.log import logger

    lines: list[str] = []
    # READ BY LABEL, NOT BY LEVEL. `level=0` takes everything loguru emits and `filter` keeps the rows
    # carrying the ledger's own marker, so what a test sees is defined by `degraded[...]` — the thing
    # the ledger IS — and not by the severity whoever wrote the emitter happened to pick. Level-gated
    # at WARNING this sink was a second, undeclared definition of the record: one `log.info` in any of
    # the ~60 emitters and `"that_kind" in degradations` reads False, `kinds()` omits it, and every
    # assertion about it passes while asserting nothing. Nothing enforces the level at the emitters,
    # so nothing may depend on it here.
    #
    # `format="{message}"` keeps the assertion about what the code SAID, not about how the run's
    # formatter is configured — the file sink's timestamp and colour tags would otherwise be part of
    # every match.
    sink = logger.add(
        lambda m: lines.append(m.record["message"]),
        level=0,
        filter=lambda r: "degraded[" in r["message"],
        format="{message}",
    )
    try:
        yield Degradations(lines)
    finally:
        logger.remove(sink)
