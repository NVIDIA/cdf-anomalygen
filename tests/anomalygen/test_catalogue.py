# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The weights catalogue, and that its readers really do read it.

The interesting tests here are the cross-language ones. Everything else is a parser, and a parser
that agrees with itself proves nothing: what this has to establish is that the shell provisioner,
the shell preflight and the Python preflight are looking at the same table -- which is the property
four hand-maintained copies did not have.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from anomalygen import catalogue, serving
from anomalygen.configs.texture.constants import BASE_CHECKPOINT_PATHS

_DOWNLOAD = catalogue.CATALOGUE.parent / "download_checkpoints.sh"


def _sh(snippet: str) -> str:
    """Run ``snippet`` with the catalogue sourced, and return its stdout."""
    out = subprocess.run(
        ["bash", "-c", f'source "{catalogue.CATALOGUE}"\n{snippet}'],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


# ── the two readers see one table ─────────────────────────────────────────────


def test_the_group_list_is_the_one_the_provisioner_offers():
    """``--list`` is what a caller reads to choose groups; asking for one it rejects is the bug."""
    listed = subprocess.run(["bash", str(_DOWNLOAD), "--list"], capture_output=True, text=True, check=True)
    assert listed.stdout.split() == list(catalogue.GROUPS)


@pytest.mark.parametrize("weight", [w for w in catalogue.WEIGHTS if w.kind != "hf"])
def test_shell_and_python_resolve_the_same_location(weight):
    assert _sh(f"w_location {weight.group}") == weight.location


@pytest.mark.parametrize("weight", [w for w in catalogue.WEIGHTS if w.kind == "hf"])
def test_shell_and_python_resolve_the_same_probes(weight):
    assert _sh(f"w_probes {weight.location}").split() == list(weight.probes)


_ONE_ROW_HF = [w for w in catalogue.WEIGHTS if w.kind == "hf" and len(catalogue.entries(w.group)) == 1]


@pytest.mark.parametrize("weight", _ONE_ROW_HF, ids=lambda w: w.group)
def test_a_hub_cache_group_answers_for_its_repo_id(weight):
    """Only for a group with one row. The guardrail holds two repositories, so w_repo refuses it for
    the same reason w_location does and both scripts name that pair literally."""
    assert _sh(f"w_repo {weight.group}") == weight.location


@pytest.mark.parametrize("group", sorted({w.group for w in catalogue.WEIGHTS if w.kind != "hf"}))
def test_a_path_addressed_group_refuses_to_pose_as_a_repo_id(group):
    """A location is a path under CKPT_DIR and may merely LOOK like a repo id.

    Three of them were being read as one with `dirname`, which happened to be right for those three.
    The catalogue's own cradio row is the counterexample: `nvidia/C-RADIO-V3/...` on disk against
    `nvidia/C-RADIOv3-B` on the Hub. The next group added would have been silently wrong.
    """
    out = subprocess.run(
        ["bash", "-c", f'source "{catalogue.CATALOGUE}"\nw_repo {group}'],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0 and "not a repo id" in out.stderr


@pytest.mark.parametrize("accessor", ["w_location", "w_repo", "w_probes"])
def test_asking_for_something_the_catalogue_lacks_stops_the_caller(accessor):
    """Under `set -e`, and NOT wrapped.

    `X="$(dirname "$(w_location typo)")"` exits 0 with X="." -- the lookup fails and prints, but
    dirname is handed the empty string and answers successfully, so the script carries on and runs
    `hf download .`. Verified; the callers resolve each location into a variable of its own now.
    """
    script = f'set -euo pipefail\nsource "{catalogue.CATALOGUE}"\nX="$({accessor} no-such-group)"\necho REACHED'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode != 0 and "REACHED" not in out.stdout


def test_a_failed_lookup_cannot_be_ignored_by_any_caller():
    """Either the script aborts on its own (`set -e`) or every lookup carries `|| exit`.

    preflight_env_ckpt.sh has no `-e` on purpose -- it collects every failure and reports them
    together -- so a bare `X="$(w_location ...)"` left X empty and carried on, and
    `check_weights_dir "${CKPT_DIR}/"` then found the first >=100MB file anywhere under checkpoints/
    and printed [pass] for a weight that was missing. Verified: it really did print that.
    """
    for script in (_DOWNLOAD, catalogue.CATALOGUE.parent / "preflight_env_ckpt.sh"):
        text = script.read_text()
        aborts = re.search(r"^set -[a-z]*e[a-z]*\b", text, re.M)
        for n, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or not re.search(r"\$\(w_(location|repo) ", stripped):
                continue
            assert aborts or stripped.endswith("|| exit 2"), (
                f"{script.name}:{n} runs without `set -e` and does not check the lookup: {stripped}"
            )


def test_no_caller_wraps_a_catalogue_lookup_in_another_command():
    """The wrapping is what swallows the failure, so the shape is what the test forbids."""
    for script in (_DOWNLOAD, catalogue.CATALOGUE.parent / "preflight_env_ckpt.sh"):
        for n, line in enumerate(script.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):  # the comment explaining the shape is not the shape
                continue
            if "$(w_location" in line or "$(w_repo" in line:
                assert not re.search(r"\$\((?!w_location|w_repo)[a-z]+ .*\$\(w_(location|repo)", stripped), (
                    f"{script.name}:{n} wraps a catalogue lookup, so a failing one exits 0: {stripped}"
                )


# ── the Python readers take their facts from it ───────────────────────────────


def test_every_group_required_weights_can_name_exists():
    """The check and the provisioning have to agree on the group names, or prepare asks for nothing.

    Read off the catalogue-driven helper rather than by enumerating configs: ``_needed`` is the only
    way a group reaches ``required_weights`` now, and it cannot produce one the catalogue lacks.
    """
    from anomalygen.scripts import run_pipeline as rp

    for group in catalogue.GROUPS:
        label, target, named = rp._needed("x", group, catalogue.entries(group)[0].location)
        assert named == group
        assert target


def test_the_serving_recipe_takes_its_repo_and_probes_from_its_catalogue_group():
    """``hf_probe`` and the script's MODEL used to be further copies of the catalogue row; the script
    names a group now and both come from it."""
    for recipe in (serving.recipe(n) for n in serving.recipe_names()):
        row = catalogue.entry(recipe.group)
        assert row.kind == "hf" and recipe.hf_repo == row.location
        assert catalogue.hf_probes(recipe.hf_repo), f"{recipe.script} serves {recipe.hf_repo}, not in the catalogue"


def test_the_training_constant_agrees_with_the_catalogue():
    """``BASE_CHECKPOINT_PATHS`` is the training side's copy and cannot move here, so pin the two."""
    for size, path in BASE_CHECKPOINT_PATHS.items():
        assert path == f"checkpoints/{catalogue.entry(catalogue.base_group(size)).location}"


# ── base_group: the drift the review found ────────────────────────────────────


@pytest.mark.parametrize("size", ["nano", "Nano", "NANO", " edge "])
def test_a_model_size_is_normalised_before_it_becomes_a_group(size):
    """``f"base-{size}"`` off an unnormalised recipe field asked the script for ``base-Nano``.

    The script rejects an unknown group, so prepare provisioned nothing and the run failed on the
    30 GB the preflight had just named -- with the config looking perfectly reasonable.
    """
    assert catalogue.base_group(size) in catalogue.GROUPS


def test_an_unknown_model_size_says_so_here():
    with pytest.raises(KeyError, match="known sizes"):
        catalogue.base_group("huge")


# ── the parser ────────────────────────────────────────────────────────────────


def test_a_group_holding_several_rows_is_not_resolved_to_one():
    """The guardrail is two repositories; answering with either would check half of it."""
    assert len(catalogue.entries("guardrail")) == 2
    with pytest.raises(KeyError, match="not one"):
        catalogue.entry("guardrail")
    assert catalogue.entry("guardrail", "Qwen/Qwen3Guard-Gen-0.6B").probes == ("model.safetensors",)


def test_an_unknown_group_is_an_error_not_an_empty_answer():
    with pytest.raises(KeyError, match="known groups"):
        catalogue.entries("no-such-group")


def test_hf_probes_are_only_read_off_hub_cache_rows():
    """A path-addressed weight has no probes, and must not answer as though it did."""
    assert catalogue.hf_probes("facebook/dinov2-large") == ()


@pytest.mark.parametrize(
    "row, why",
    [
        ('  "a|dir|b|c|d"\n', "cannot read catalogue row"),
        ('  "a|blob|b|"\n', "not one of"),
        ("  a|dir|b|\n", "cannot read catalogue row"),
    ],
)
def test_a_row_that_cannot_be_read_is_refused(row, why):
    """Silently skipping one would drop a weight from every check that reads this."""
    with pytest.raises(RuntimeError, match=why):
        catalogue._parse(f"WEIGHTS=(\n{row})\n")


def test_rows_outside_the_array_are_not_picked_up():
    """The file's header carries a worked example; matching it would invent a twelfth entry."""
    text = '# "example|dir|not-real|"\nWEIGHTS=(\n  "a|dir|b|"\n)\n# "trailing|file|nope|"\n'
    assert [w.location for w in catalogue._parse(text)] == ["b"]


# ── the air-gapped image ──────────────────────────────────────────────────────

# The one row the air-gapped stage deliberately leaves out, with the reason the Dockerfile gives.
# Listed rather than inferred: "not baked" and "forgotten" look identical in a Dockerfile, and the
# difference is an image whose text2roi dies on its first fetch on a node that cannot fetch.
_NOT_BAKED = {"cosmos-vlm": "~31.5 GB HF-format Nano copy; in-image preflight reports it missing"}

_DOCKERFILE = catalogue.CATALOGUE.parent.parent / "docker" / "Dockerfile"
_BAKED = "/workspace/cdf-anomalygen/checkpoints/"


def _baked() -> list:
    """The checkpoint paths the air-gapped stage COPYs in, relative to ``checkpoints/``.

    Continuations are joined first, and only a destination on a real ``COPY --from=ckpts`` counts.
    Scanning raw lines for the prefix was the first version of this, and it read the second half of
    a two-line COPY as a baked path in its own right -- so deleting the COPY it belonged to left the
    test green, which is the one thing it exists to catch.
    """
    text = _DOCKERFILE.read_text().replace("\\\n", " ")
    baked = []
    for line in text.splitlines():
        if not line.startswith("COPY --from=ckpts "):
            continue
        baked += [w[len(_BAKED) :].rstrip("/") for w in line.split() if w.startswith(_BAKED)]
    return baked


@pytest.mark.parametrize("weight", catalogue.WEIGHTS, ids=lambda w: f"{w.group}:{w.location}")
def test_every_catalogued_weight_is_baked_into_the_airgapped_image(weight):
    """A COPY cannot read the catalogue, so this is the only thing holding the two together.

    An air-gapped node cannot fetch what its image left out, and the failure lands at the first load
    on a customer's machine rather than at build time.
    """
    if weight.group in _NOT_BAKED:
        pytest.skip(_NOT_BAKED[weight.group])
    if weight.kind == "hf":
        wanted = f"hf/models--{weight.location.replace('/', '--')}"
    else:
        wanted = weight.location
    # A COPY of the parent directory carries the file, so a prefix counts.
    assert any(wanted == b or wanted.startswith(f"{b}/") for b in _baked()), (
        f"{weight.location} is in the catalogue but nothing under {_BAKED} covers it"
    )


def test_the_exclusion_list_only_names_real_groups():
    """A stale exclusion would silently excuse whatever group later took the name."""
    assert set(_NOT_BAKED) <= set(catalogue.GROUPS)
