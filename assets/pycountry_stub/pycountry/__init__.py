# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-only stub for the real ``pycountry`` package.

Exposes the five ISO databases ``pydantic_extra_types`` reads (``countries``, ``currencies``,
``languages``, ``language_families``, ``scripts``) as empty collections: iteration yields nothing,
``get`` finds nothing, ``lookup`` raises. Enough for the modules to import; every code that
validates against them is rejected.
"""


class _EmptyDatabase:
    """Stands in for a ``pycountry.db.Database``: iterable, searchable, always empty."""

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def get(self, **kwargs):
        return None

    def lookup(self, value):
        raise LookupError(f"pycountry stub holds no data; cannot look up {value!r}")


countries = _EmptyDatabase()
currencies = _EmptyDatabase()
languages = _EmptyDatabase()
language_families = _EmptyDatabase()
scripts = _EmptyDatabase()
