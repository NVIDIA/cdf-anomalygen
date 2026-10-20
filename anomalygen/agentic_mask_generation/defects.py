# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distilled defect-knowledge cards (the INJECT side of the defect KB).

A card is curated markdown at ``<cards_dir>/<type>/<variant>.md``, injected into the proposer
(``card()``) and the authoritative judge (``docs()``). Frozen-model / in-context only; an un-filled
schema template counts as absent (``None``)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Protocol

# A card still carrying the distillation SCHEMA (HTML-comment instructions / empty sections) is unfilled.
_TEMPLATE_MARK = "schema template"

# Shared anatomy note appended to every real card, when curated alongside the cards.
_GENERAL_NOTE = ("_general", "lead_solder_anatomy.md")


def _strip_template(md: str) -> str:
    """Drop the HTML-comment schema instructions and blank lines, leaving substantive prose."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    return "\n".join(ln for ln in md.splitlines() if ln.strip()).strip()


class DefectLibrary(Protocol):
    """Distilled defect knowledge: ``card`` is the proposer scope, ``docs`` the judge scope, None if absent."""

    def card(self, defect_type: str) -> Optional[str]: ...

    def docs(self, defect_type: str) -> Optional[str]: ...


class LocalCards:
    """Reads distilled cards from ``<cards_dir>/<defect_type>/<variant>.md``. The card-file stem is
    written by ``card_distill.card_variant``, and a disagreement there yields a silently EMPTY library,
    so the two sides must agree: each model gets its own card, and the writer and the reader
    agree on the name."""

    def __init__(self, cards_dir: str, *, variant: str = "card") -> None:
        self._root = Path(cards_dir)
        self._variant = variant

    def _load(self, defect_type: str) -> Optional[str]:
        p = self._root / defect_type / f"{self._variant}.md"
        if not p.is_file():
            return None
        txt = p.read_text(errors="ignore")
        body = _strip_template(txt)
        # 120 chars: enough to be prose rather than a heading and a blank section.
        if _TEMPLATE_MARK in txt or len(body) <= 120:
            return None
        parts = [body]
        gen = self._root.joinpath(*_GENERAL_NOTE)
        if gen.is_file():
            parts.append("GENERAL REFERENCE:\n" + _strip_template(gen.read_text(errors="ignore")))
        return "\n\n".join(parts)

    def card(self, defect_type: str) -> Optional[str]:
        return self._load(defect_type)

    def docs(self, defect_type: str) -> Optional[str]:
        return self._load(defect_type)


class NoDefects:
    """A library with nothing in it."""

    def card(self, defect_type: str) -> Optional[str]:
        return None

    def docs(self, defect_type: str) -> Optional[str]:
        return None


def from_dir(cards_dir: Optional[str], *, variant: str = "card") -> DefectLibrary:
    """The one way a caller picks an implementation: no directory configured means no cards."""
    if cards_dir is None:
        return NoDefects()
    return LocalCards(cards_dir, variant=variant)
