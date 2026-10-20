# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one prepare script whose dataset cannot be fully downloaded.

Its masks and defect_spec come from Hugging Face; its images come from a Roboflow export that a
human downloads through a browser. Everything here is about what happens when that file is not
given -- which used to be "write the masks and exit 0", a success that produced a dataset with no
images in it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from anomalygen.scripts.datasets import prepare_phone_screen_defect as script
from anomalygen.staging import usecase


def test_a_run_with_no_zip_says_the_images_are_not_downloadable(tmp_path, capsys):
    """`usecase://phone_screen` used to fetch cleanly and leave Phone/clean_image/ absent. What
    surfaced was an override refusing a path "not in the fetched input", or an AMP run with no
    exemplar -- neither of which mentions a zip."""
    with pytest.raises(SystemExit) as exc:
        script.main(["--output_dir", str(tmp_path)])

    message = str(exc.value)
    assert script.ROBOFLOW_URL in message, "the message has to say where the file comes from"
    assert "--zip" in message and "--masks_only" in message
    assert not list(tmp_path.iterdir()), "nothing may be written before the refusal"


def test_the_refusal_comes_before_the_hugging_face_check(tmp_path, monkeypatch):
    """A missing zip needs no token to diagnose. Reporting the token first sent someone to fix a
    credential and meet the real problem on the next run."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-token-here"))
    called = []
    monkeypatch.setattr(script, "preflight", lambda args: called.append("preflight"))

    with pytest.raises(SystemExit) as exc:
        script.main(["--output_dir", str(tmp_path)])
    assert "Roboflow" in str(exc.value) or script.ROBOFLOW_URL in str(exc.value)
    assert called == [], "preflight must not have run"


def test_masks_only_is_the_way_to_ask_for_the_incomplete_dataset(tmp_path, monkeypatch):
    """The escape hatch stays, but it has to be asked for: the result has no images, so nothing can
    be generated from it."""
    ran = []
    monkeypatch.setattr(script, "preflight", lambda args: ran.append("preflight"))
    monkeypatch.setattr(script, "download", lambda args: ran.append("download"))
    monkeypatch.setattr(script, "post_process", lambda args, raw: ran.append("post_process"))

    script.main(["--output_dir", str(tmp_path), "--masks_only"])
    assert ran == ["preflight", "download", "post_process"]


def test_a_zip_is_accepted_without_masks_only(tmp_path, monkeypatch):
    """The ordinary complete run must not have been made harder by the guard."""
    zip_path = tmp_path / "mobile-screen.zip"
    zip_path.write_text("not really a zip")
    monkeypatch.setattr(script, "preflight", lambda args: None)
    monkeypatch.setattr(script, "download", lambda args: None)
    seen = {}
    monkeypatch.setattr(script, "post_process", lambda args, raw: seen.update(zip=args.zip))

    script.main(["--output_dir", str(tmp_path), "--zip", str(zip_path)])
    assert seen["zip"] == str(zip_path)


# ── every shipped prepare script, not just this one ───────────────────────────


def test_this_script_rejects_an_abbreviated_flag(tmp_path):
    """These scripts are launched by `usecase://<name>?<p>=...`, where the parameter names come from
    a config -- and argparse accepts any unambiguous abbreviation of a long option, so `?output=`
    reached `--output_dir` and sent the whole dataset wherever the config said.

    A subprocess, because the property belongs to the parser as it is actually built. `--dryru` is an
    abbreviation of a real flag, so it is ACCEPTED (exit 0, nothing written) exactly when abbreviation
    is on, and unrecognised (exit 2) when it is off. One script is run this way; the check below
    covers the rest, which each cost eight seconds of imports to launch.
    """
    done = subprocess.run(
        [sys.executable, "-m", script.__name__, "--output_dir", str(tmp_path), "--dryru"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2, f"an abbreviated flag was accepted: {done.stdout[-300:]}"
    assert "unrecognized arguments" in done.stderr


@pytest.mark.parametrize("name", usecase.names())
def test_every_shipped_prepare_script_turns_abbreviation_off(name):
    """Read off what the image ships rather than listed here: a fourth script is added by adding the
    file, and it is launched the same way from the same config-supplied names. staging/usecase.py
    guards its own flag as well, but only the parser can close the class -- a flag added to a script
    later would reopen it, and every abbreviation of it with it."""
    source = (pathlib.Path(script.__file__).parent / f"prepare_{name}_defect.py").read_text()
    call = source[source.index("argparse.ArgumentParser(") :]
    call = call[: call.index("\n    )") if "\n    )" in call else call.index(")\n")]
    assert "allow_abbrev=False" in call, f"prepare_{name}_defect.py builds a parser that abbreviates"
