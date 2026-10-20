# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Hugging Face and NGC fetch sources.

No network: every test here fakes the one call that would make it, because a test that reaches a
public registry fails on an air-gapped runner and on an outage, neither of which is a defect in this
code. The live behaviour these fakes stand in for was verified against the real endpoints.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from anomalygen.staging import hub, registry, store, usecase

# ── what each scheme is allowed to be ─────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    ["s3://b/k", "msc://p/k", "hf://datasets/org/repo", "ngc://models/org/name:1.0"],
)
def test_every_fetch_scheme_is_remote(url):
    assert store.is_remote(url)


@pytest.mark.parametrize("url", ["hf://datasets/org/repo", "ngc://models/org/name:1.0"])
def test_a_registry_is_a_source_and_never_a_publish_destination(url):
    """Publishing to Hugging Face or NGC is a human decision with a licence attached to it. A config
    field typo must not be able to start one, so the write path does not accept these at all."""
    store.check_source(url)  # readable
    with pytest.raises(ValueError, match="fetch sources, not upload destinations"):
        store.check_uri(url)


# ── Hugging Face ──────────────────────────────────────────────────────────────


def test_the_repo_type_must_be_written_out(monkeypatch):
    """HfFileSystem reads a bare hf://org/repo as a MODEL repo, so a dataset given that way fails
    with 'model not found' -- pointing at a repo the user never named."""
    with pytest.raises(ValueError, match="must name the repo type"):
        hub._parse("hf://nvidia/Cosmos-AnomalyGen-PCB-Dataset")


class _FakeFS:
    """Enough of HfFileSystem to exercise the listing, faithfully on the one axis that broke it.

    find() prefixes the repo type for datasets and NOT for models -- huggingface_hub's own
    REPO_TYPES_URL_PREFIXES has no entry for "model" -- so this mirrors that table rather than
    assuming both look alike. The previous fake echoed a datasets/ prefix whatever the type, so the
    model branch was never exercised and every hf://models/ prefix fetch was broken in a way CI
    could not see.

    resolve_path is a lookup over what find() returned, not a parser: the real one makes a network
    call, and a second parser here would be a second thing to get wrong.
    """

    def __init__(self, paths, repo_type="dataset", echo_revision=True):
        self.paths = paths
        self.repo_type = repo_type
        self.echo_revision = echo_revision
        self._resolved = {}

    @staticmethod
    def _prefix(repo_type):
        return {"dataset": "datasets/", "space": "spaces/", "kernel": "kernels/"}.get(repo_type, "")

    def resolve_path(self, url):
        import types

        if url in self._resolved:
            return self._resolved[url]
        rest = url[len("hf://") :] if url.startswith("hf://") else url
        plural, _, tail = rest.partition("/")
        repo = "/".join(tail.split("/")[:2])
        path_in_repo = "/".join(tail.split("/")[2:])
        repo_id, _, revision = repo.partition("@")
        return types.SimpleNamespace(
            repo_type=plural.rstrip("s"),
            repo_id=repo_id,
            revision=revision or "main",
            path_in_repo=path_in_repo,
        )

    def find(self, path):
        import types

        # The real one would look up a repo literally named "models/nvidia" and fail. Asserting the
        # form here is what makes a wrong prefix visible: without it this fake accepts any shape and
        # the caller's mistake is invisible until a live run.
        expected = self._prefix(self.repo_type)
        assert path.startswith(expected), f"{path!r} does not start with {expected!r} for a {self.repo_type}"
        body = path[len(expected) :].partition("@")[0]
        assert body.count("/") == 1, f"{body!r} is not <org>/<repo>; a type prefix was left on it"

        root, _, rest = path.partition("@")
        rev, _, base = rest.partition("/") if "@" in path else ("", "", "")
        base = base.rstrip("/")
        repo_id = root[len(self._prefix(self.repo_type)) :]
        out = []
        for p in self.paths:
            in_repo = f"{base}/{p}" if base else p
            shown = f"{self._prefix(self.repo_type)}{repo_id}"
            if rev and self.echo_revision:
                shown += f"@{rev}"
            shown = f"{shown}/{in_repo}"
            self._resolved[shown] = types.SimpleNamespace(
                repo_type=self.repo_type, repo_id=repo_id, revision=rev or "main", path_in_repo=in_repo
            )
            out.append(shown)
        return out


def _fake_hf(monkeypatch, paths, *, sha="a" * 40, echo_revision=True, repo_type="dataset"):
    monkeypatch.setattr(hub, "_fs", lambda: _FakeFS(paths, repo_type=repo_type, echo_revision=echo_revision))
    monkeypatch.setattr(hub, "_commit", lambda resolved: sha)


@pytest.mark.parametrize("echo", [True, False])
def test_listing_survives_either_answer_shape(monkeypatch, echo):
    """The real HfFileSystem echoes the revision when asked with one and omits it when not. Which
    one it does is invisible until a tree lands flattened into a single directory."""
    _fake_hf(monkeypatch, ["Phone/mask/oil/a.png", "defect_spec.jsonl"], echo_revision=echo)
    found = dict(hub.iter_objects("hf://datasets/nvidia/Glass"))
    assert sorted(found.values()) == ["Phone/mask/oil/a.png", "defect_spec.jsonl"]
    assert all(u.startswith("hf://datasets/nvidia/Glass@" + "a" * 40 + "/") for u in found)


def test_the_tree_below_a_subpath_is_relative_to_it(monkeypatch):
    """Fetching <repo>/Phone/mask must put oil/a.png at oil/a.png, not Phone/mask/oil/a.png."""
    _fake_hf(monkeypatch, ["oil/a.png"])  # find() is asked for the subpath and answers below it
    found = dict(hub.iter_objects("hf://datasets/nvidia/Glass/Phone/mask"))
    assert list(found.values()) == ["oil/a.png"]
    assert list(found) == ["hf://datasets/nvidia/Glass@" + "a" * 40 + "/Phone/mask/oil/a.png"]


def test_hugging_face_plumbing_is_not_dataset_content(monkeypatch):
    """Every HF repo carries .gitattributes and no consumer of a defect dataset wants it."""
    _fake_hf(monkeypatch, [".gitattributes", "defect_spec.jsonl"])
    assert sorted(dict(hub.iter_objects("hf://datasets/nvidia/Glass")).values()) == ["defect_spec.jsonl"]


def test_a_listing_outside_the_prefix_is_an_error_not_a_guess(monkeypatch):
    """Placing it would need a guess, and guessing wrong flattens the tree without failing.

    Asked for one subdirectory and answered with a file outside it -- the shape that matters, since
    a fetch of the whole repo has no prefix to escape from."""

    class _Escaping(_FakeFS):
        def find(self, path):
            import types

            out = "Elsewhere/x.png"
            self._resolved[out] = types.SimpleNamespace(
                repo_type="dataset", repo_id="nvidia/Glass", revision="a" * 40, path_in_repo=out
            )
            return [out]

    monkeypatch.setattr(hub, "_fs", lambda: _Escaping([]))
    monkeypatch.setattr(hub, "_commit", lambda resolved: "a" * 40)
    with pytest.raises(RuntimeError, match="not under"):
        list(hub.iter_objects("hf://datasets/nvidia/Glass/Phone/mask"))


@pytest.mark.parametrize(
    ("repo_type", "url"),
    [("dataset", "hf://datasets/nvidia/Glass"), ("model", "hf://models/nvidia/Tuned")],
)
def test_a_model_repo_lists_the_same_as_a_dataset_repo(monkeypatch, repo_type, url):
    """find() omits the repo-type prefix for models and keeps it for datasets. Reconstructing the
    prefix by hand made every hf://models/ prefix fetch fail its own "not under" guard on the first
    entry -- the fine-tuned-checkpoint case the module docstring names -- while datasets worked."""
    monkeypatch.setattr(hub, "_fs", lambda: _FakeFS(["a/b.bin", "c.json"], repo_type=repo_type))
    monkeypatch.setattr(hub, "_commit", lambda resolved: "a" * 40)
    found = dict(hub.iter_objects(url))
    assert sorted(found.values()) == ["a/b.bin", "c.json"]
    assert all(u.startswith(f"{url}@" + "a" * 40 + "/") for u in found), found


def test_the_listing_is_pinned_even_when_the_url_spelled_the_branch(monkeypatch):
    """A source written `@main` carries that string, and the resolved-path object prefers it over
    the revision field -- so replacing the revision on it listed main instead of the sha just
    resolved, quietly undoing the pin this module resolves a sha for."""
    seen = []

    class _Recording(_FakeFS):
        def find(self, path):
            seen.append(path)
            return super().find(path)

    monkeypatch.setattr(hub, "_fs", lambda: _Recording(["x.png"], repo_type="dataset"))
    monkeypatch.setattr(hub, "_commit", lambda resolved: "b" * 40)
    list(hub.iter_objects("hf://datasets/nvidia/Glass@main/Phone"))
    assert seen == ["datasets/nvidia/Glass@" + "b" * 40 + "/Phone"], seen


def test_models_carry_no_type_prefix_and_datasets_do(monkeypatch):
    """Pinned against huggingface_hub's own table rather than a guess: REPO_TYPES_URL_PREFIXES has
    entries for dataset, space and kernel, and none for model."""
    import types

    ds = types.SimpleNamespace(repo_type="dataset", repo_id="org/repo")
    md = types.SimpleNamespace(repo_type="model", repo_id="org/repo")
    assert hub._fs_path(ds, "abc", "sub") == "datasets/org/repo@abc/sub"
    assert hub._fs_path(md, "abc", "sub") == "org/repo@abc/sub"
    assert hub._fs_path(md, "abc", "") == "org/repo@abc"


def test_an_unpinned_source_reports_the_commit_it_resolved(monkeypatch):
    """It cannot refuse one, so the log is the only reproducibility record the run leaves.

    Asserted on the logger rather than through caplog: this package logs via cosmos_framework's
    loguru sink, which does not pass through the stdlib logging caplog hooks into."""
    import types

    warned = []
    monkeypatch.setattr(hub.log, "warning", warned.append)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.repo_info",
        lambda self, repo_id, repo_type=None, revision=None: types.SimpleNamespace(sha="b" * 40),
    )
    resolved = types.SimpleNamespace(repo_type="dataset", repo_id="nvidia/Glass", revision="main")
    assert hub._commit(resolved) == "b" * 40
    assert "b" * 40 in warned[0]
    assert "branch, not a commit" in warned[0]


# ── NGC ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("ngc://models/nvidia/team/name:1.0", ("models", "nvidia/team/name", "1.0", "")),
        ("ngc://models/nvidia/name:1.0", ("models", "nvidia/name", "1.0", "")),
        ("ngc://resources/o/t/n:v1.4.1/a/b.yaml", ("resources", "o/t/n", "v1.4.1", "a/b.yaml")),
    ],
)
def test_the_url_splits_into_what_the_rest_path_needs(url, expected):
    assert registry._parse(url) == expected


# The endpoint is chosen by whether there is a key, because NGC has two and neither serves both
# cases. Only the public one shipped, so no private artifact could ever be fetched -- and nothing
# caught it, because every test and every e2e run used a PUBLIC artifact, where that endpoint is
# right and the Authorization header is ignored. These are the four measurements that separate them.
_NGC_PCB = "ngc://models/myorg/myteam/cosmos-anomalygen-pcb:1.1"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # org + team
        (_NGC_PCB, "/v2/org/myorg/team/myteam/models/cosmos-anomalygen-pcb/versions/1.1"),
        # org only -- NGC allows an artifact directly under an org, so the team segment disappears
        ("ngc://models/someorg/somemodel:2.0", "/v2/org/someorg/models/somemodel/versions/2.0"),
        # a file inside the version addresses the same endpoint; the path is returned separately
        (
            f"{_NGC_PCB}/iter_000014000.pt",
            "/v2/org/myorg/team/myteam/models/cosmos-anomalygen-pcb/versions/1.1",
        ),
    ],
    ids=["org-team", "org-only", "file-in-version"],
)
def test_a_key_addresses_the_org_scoped_endpoint(url, expected, monkeypatch):
    """The only one that serves a private artifact. Measured against the real API:

        404  /v2/models/myorg/myteam/cosmos-anomalygen-pcb/versions/1.1/files
        200  /v2/org/myorg/team/myteam/models/...  (same artifact, same key)

    404 rather than 403, because answering "forbidden" would confirm the name exists -- so the one
    code means both "no such artifact" and "not yours", and the error sends the reader to check a
    spelling that was correct.
    """
    monkeypatch.setenv("NGC_API_KEY", "nvapi-pretend")
    assert registry._base(url)[0].endswith(expected)


def test_without_a_key_the_public_catalogue_is_addressed_instead(monkeypatch):
    """NOT simply "use the org-scoped one always", which is what I wrote first.

    The org-scoped endpoint answers 401 with no credentials even for a PUBLIC artifact, so that
    version would have broken every keyless fetch -- the e2e run that takes a public NGC resource,
    and every test in this file:

        200  /v2/models/nvidia/tao/pretrained_classification/versions/resnet18/files   (no key)
        401  /v2/org/nvidia/team/tao/resources/cv_samples/versions/v1.4.1/files        (no key)
    """
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    base = registry._base("ngc://resources/nvidia/tao/cv_samples:v1.4.1/action_recognition_net")[0]
    assert base.endswith("/v2/resources/nvidia/tao/cv_samples/versions/v1.4.1")
    assert "/org/" not in base


def test_the_two_endpoints_are_not_the_same_url(monkeypatch):
    """A regression guard with no opinion about which is which: if a later simplification collapses
    them into one form, one of the two cases stops working and nothing else here would say so."""
    monkeypatch.setenv("NGC_API_KEY", "nvapi-pretend")
    with_key = registry._base(_NGC_PCB)[0]
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    without = registry._base(_NGC_PCB)[0]
    assert with_key != without


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("ngc://nvidia/team/name:1.0", "must name the artifact type"),
        ("ngc://models/nvidia/team/name", "names no version"),
        ("ngc://models/name:1.0", "segment"),
        ("ngc://models/nvidia/name:", "empty version"),
    ],
)
def test_a_url_that_cannot_address_an_artifact_is_rejected(url, match):
    with pytest.raises(ValueError, match=match):
        registry._parse(url)


def _paged(monkeypatch, pages, total):
    """Serve ``pages`` of file paths the way NGC does, one request per page."""
    seen = []

    class _Response:
        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

    def _get(url, **kwargs):
        page = int(kwargs.get("params", {}).get("page-number", 0))
        seen.append(page)
        nxt = page + 1 if page + 1 < len(pages) else None
        return _Response(
            {
                "modelFiles": [{"path": p} for p in pages[page]],
                "modelVersion": {"totalFileCount": total},
                "paginationInfo": {"index": page, "nextPage": nxt},
            }
        )

    monkeypatch.setattr(registry, "_get", _get)
    return seen


def test_every_page_is_read_not_just_the_first(monkeypatch):
    """NGC answers with 100 files and a paginationInfo block whether or not more exist. A
    single-request implementation returns a short list and reports success."""
    seen = _paged(monkeypatch, [[f"f{i}" for i in range(100)], ["f100", "f101"]], total=102)
    assert len(registry._files("ngc://models/nvidia/n:1.0")) == 102
    assert seen == [0, 1]


def test_a_lost_page_fails_rather_than_fetching_a_partial_tree(monkeypatch):
    """A tree that is 100 files instead of 205 does not fail downstream -- it quietly generates from
    less than it was given, which is why the version's own count is checked against."""
    _paged(monkeypatch, [["a", "b"]], total=205)
    with pytest.raises(RuntimeError, match="A page was lost"):
        registry._files("ngc://models/nvidia/n:1.0")


def test_a_subpath_selects_on_the_separator(monkeypatch):
    """NGC versions are flat path lists, so a bare prefix match would pull in a sibling: asking for
    'mask' must not also bring 'mask_v2/...'."""
    _paged(monkeypatch, [["mask/a.png", "mask_v2/b.png", "mask/c/d.png"]], total=3)
    found = {rel: url for url, rel in registry.iter_objects("ngc://models/nvidia/n:1.0/mask")}
    assert sorted(found) == ["a.png", "c/d.png"]
    assert found["a.png"] == "ngc://models/nvidia/n:1.0/mask/a.png"


def test_the_api_key_is_read_from_the_environment_only(monkeypatch):
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    assert registry._headers() == {}
    monkeypatch.setenv("NGC_API_KEY", "nvapi-xyz")
    assert registry._headers() == {"Authorization": "Bearer nvapi-xyz"}


# ── the publish destination, checked before the run rather than after ─────────


def test_an_empty_prefix_is_reachable(monkeypatch):
    """Publishing into a prefix that does not exist yet is the normal case, not an error -- every
    first run of a new output location would otherwise be rejected."""
    monkeypatch.setattr(store, "configure", lambda: None)
    fake = type("msc", (), {"list": staticmethod(lambda *a, **k: iter([]))})
    monkeypatch.setitem(__import__("sys").modules, "multistorageclient", fake)
    store.check_reachable("s3://bucket/brand/new/")


def test_an_unreachable_destination_is_rejected_with_what_it_could_mean(monkeypatch):
    """The message must not lead with credentials. A run that published seven batches and then met a
    typo'd bucket was told to check its AWS keys, which were fine."""
    monkeypatch.setattr(store, "configure", lambda: None)

    def _boom(*a, **k):
        raise RuntimeError("NoSuchBucket: The specified bucket does not exist")

    fake = type("msc", (), {"list": staticmethod(_boom)})
    monkeypatch.setitem(__import__("sys").modules, "multistorageclient", fake)
    with pytest.raises(RuntimeError) as exc:
        store.check_reachable("s3://nope/results/")
    message = str(exc.value)
    assert "NoSuchBucket" in message, "the store's own words must reach the caller"
    assert "does not exist" in message and "endpoint is wrong" in message, "and what it could mean"
    assert message.index("bucket or prefix does not exist") < message.index("no credentials at all"), (
        "credentials are one cause among four, not the headline"
    )


# ── what a fetch must refuse ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rel", "why"),
    [
        ("/etc/cron.d/job", "an S3 key with a double slash, which S3 allows"),
        ("../../etc/cron.d/job", "an entry containing .., which NGC path lists allow"),
        ("a/../../../outside.txt", "a climb hidden mid-path"),
    ],
)
def test_a_listing_cannot_place_a_file_outside_the_staging_directory(tmp_path, monkeypatch, rel, why):
    """`Path(staged) / "/etc/cron.d/job"` IS `/etc/cron.d/job` -- the join silently discards the left
    side. The file would survive the swap, outside the tree, and the run would report it fetched."""
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}x", rel)]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("payload"))
    with pytest.raises(RuntimeError, match="does not land inside"):
        staging.fetch(["s3://b/x/"], tmp_path / "ds")
    assert not pathlib.Path("/etc/cron.d/job").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_a_destination_this_run_did_not_write_is_not_replaced(tmp_path, monkeypatch):
    """`input: /data/tht` plus a staging.fetch.input downloads the overlay elsewhere, then deletes
    the dataset at /data/tht and puts the download in its place. The same-path guard does not catch
    it: the source and the destination are different paths."""
    from anomalygen import staging

    dataset = tmp_path / "tht"
    (dataset / "T" / "clean_image").mkdir(parents=True)
    (dataset / "T" / "clean_image" / "j1.png").write_text("five thousand of these")

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("new"))

    with pytest.raises(RuntimeError, match="no previous fetch wrote it"):
        staging.fetch(["s3://b/extra/"], dataset)
    assert (dataset / "T" / "clean_image" / "j1.png").read_text() == "five thousand of these"


def test_a_destination_a_previous_fetch_wrote_is_replaced(tmp_path, monkeypatch):
    """The guard must not break the re-run it is meant to allow: a second run over the same output
    replaces what the first one fetched."""
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("new"))

    dest = tmp_path / "ds"
    staging.fetch(["s3://b/x/"], dest)
    assert (dest / staging.FETCH_MARKER).exists(), "the first fetch must claim the directory"
    staging.fetch(["s3://b/x/"], dest)
    assert (dest / "a.png").read_text() == "new"


def _one_file(monkeypatch, staging, body="fetched"):
    """A source that names a single object, so fetch takes its single-file branch."""
    monkeypatch.setattr(staging, "is_object", lambda uri: True)
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text(body))


def test_a_single_file_destination_somebody_else_owns_is_refused(tmp_path, monkeypatch):
    """`recipe`, `checkpoint` and `amg_model_registry` are FILE destinations, and the guard only ran
    on the prefix path -- the single-object branch returned three lines above it. So a config that
    named an operator's checkpoint and a fetch for it replaced the file with one log line."""
    from anomalygen import staging

    _one_file(monkeypatch, staging)
    theirs = tmp_path / "iter_000035000.pt"
    theirs.write_text("the fine-tuned checkpoint, 30 GB of it")

    with pytest.raises(RuntimeError, match="no previous fetch wrote it"):
        staging.fetch("s3://b/ckpt.pt", theirs)
    assert theirs.read_text() == "the fine-tuned checkpoint, 30 GB of it"


def test_a_single_file_a_previous_fetch_wrote_is_replaced(tmp_path, monkeypatch):
    """The marker for a file is a sidecar, since a file cannot carry one inside itself -- and the
    re-run it allows is the whole point of the marker."""
    from anomalygen import staging

    _one_file(monkeypatch, staging, "first")
    dest = tmp_path / "r.yaml"
    staging.fetch("s3://b/r.yaml", dest)
    assert dest.read_text() == "first"
    assert staging._marker_for(dest, one_file=True).exists()

    _one_file(monkeypatch, staging, "second")
    staging.fetch("s3://b/r.yaml", dest)
    assert dest.read_text() == "second"


def test_a_prefix_fetch_onto_a_file_says_so_instead_of_dying_in_rmtree(tmp_path, monkeypatch):
    """It passed the guard -- not a directory, so nothing to refuse -- and raised a bare
    NotADirectoryError out of shutil.rmtree, naming neither the path nor the config key."""
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("new"))

    dest = tmp_path / "ds"
    dest.write_text("a file, where a dataset directory was meant")
    with pytest.raises(RuntimeError, match="is a file, but this fetch lists a prefix"):
        staging.fetch(["s3://b/x/"], dest)
    assert dest.read_text() == "a file, where a dataset directory was meant"


def test_a_single_file_fetch_onto_a_directory_says_so(tmp_path, monkeypatch):
    """The other half of the kind check: _copy_one would have raised IsADirectoryError."""
    from anomalygen import staging

    _one_file(monkeypatch, staging)
    dest = tmp_path / "ds"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="is a directory, but this fetch names a single file"):
        staging.fetch("s3://b/r.yaml", dest)


def test_a_directory_can_be_adopted_on_purpose(tmp_path, monkeypatch):
    """A PVC whose filesystem left a lost+found, or a tree fetched before the marker existed, would
    otherwise be unusable until deleted by hand. Creating the marker is the operator saying the
    directory is theirs to replace -- an explicit assertion, not a config flag that would have to
    mean the same thing."""
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("new"))

    dest = tmp_path / "pvc"
    (dest / "lost+found").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="adopt it deliberately"):
        staging.fetch(["s3://b/x/"], dest)

    staging._marker_for(dest, one_file=False).touch()
    assert staging.fetch(["s3://b/x/"], dest) == 1


def test_a_cancellation_during_the_swap_never_leaves_nothing(tmp_path, monkeypatch):
    """The entrypoint turns SIGTERM into SystemExit, so it can arrive anywhere in here.

    The old tree was deleted BEFORE the new one was renamed in, and the finally then removed the
    download -- so a cancellation in that window destroyed both copies of a dataset it had just
    spent an hour fetching. The docstring promised the opposite.
    """
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("new"))

    dest = tmp_path / "ds"
    staging.fetch(["s3://b/x/"], dest)  # a first fetch, so the marker lets the second replace it
    assert (dest / "a.png").read_text() == "new"

    # Cancelled at the moment the old tree has been moved aside and the new one not yet renamed in.
    real_replace = pathlib.Path.replace
    calls = []

    def cancel_on_second(self, target):
        calls.append(self)
        if len(calls) == 2:
            raise SystemExit("cancelled, as SIGTERM does")
        return real_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", cancel_on_second)
    with pytest.raises(SystemExit):
        staging.fetch(["s3://b/x/"], dest)
    monkeypatch.undo()

    assert dest.is_dir(), "the destination is gone entirely"
    assert (dest / "a.png").read_text() == "new", "the old contents were not restored"
    assert not list(dest.parent.glob(".ds.replaced")), "the aside copy was left behind"


def test_an_aside_copy_left_by_a_hard_kill_is_reclaimed_not_orphaned(tmp_path, monkeypatch):
    """The recovery below only runs when the process reaches its own finally.

    A pod eviction, or the SIGKILL after an orchestrator's grace period, reaches neither -- so the
    old tree sits in .<name>.replaced with no destination beside it, and the first version of this
    fix skipped the whole branch (it was nested under `destination.exists()`), renamed the new tree
    into place and orphaned a 5k-image dataset on the volume for good.
    """
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("new"))

    dest = tmp_path / "ds"
    # Exactly what a kill between the two renames leaves: no destination, the old tree aside.
    aside = dest.with_name(".ds.replaced")
    aside.mkdir()
    (aside / "a.png").write_text("the tree that run was replacing")
    (aside / staging.FETCH_MARKER).touch()

    assert staging.fetch(["s3://b/x/"], dest) == 1
    assert (dest / "a.png").read_text() == "new"
    assert not aside.exists(), "the old tree was left orphaned on the volume"


def test_an_aside_copy_from_an_interrupted_run_does_not_block_the_next(tmp_path, monkeypatch):
    """A run killed between the two renames can leave one. The next fetch owns it, not the operator."""
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("new"))

    dest = tmp_path / "ds"
    staging.fetch(["s3://b/x/"], dest)
    leftover = dest.with_name(".ds.replaced")
    leftover.mkdir()
    (leftover / "stale.png").write_text("from a run that died mid-swap")

    assert staging.fetch(["s3://b/x/"], dest) == 1
    assert not leftover.exists()


def test_an_empty_destination_is_fine(tmp_path, monkeypatch):
    """An empty directory is not somebody's data, and orchestrators pre-create mount points."""
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text("new"))

    dest = tmp_path / "ds"
    dest.mkdir()
    assert staging.fetch(["s3://b/x/"], dest) == 1


# ── use-case sources ──────────────────────────────────────────────────────────
#
# A use-case is RUN to produce its dataset, so the tests below stub the script rather than the
# network: the real ones download tens of megabytes, and what has to be pinned here is the contract
# between fetch and a prepare script -- name resolution, the subprocess, and what an empty or failed
# run means -- not the datasets themselves.


def _fake_script(monkeypatch, tmp_path, *, writes=None, code=0):
    """Stand in for `python -m ...prepare_<name>_defect --output_dir X`, writing `writes` into X."""
    calls = []

    def run(cmd, on_line, **kw):
        argv = [str(c) for c in cmd]
        calls.append(argv)
        out = pathlib.Path(argv[argv.index("--output_dir") + 1])
        for rel in writes or []:
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")
        on_line("Done: pretend\n")
        return code

    monkeypatch.setattr(usecase.childproc, "stream", run)
    monkeypatch.setattr(usecase, "names", lambda: ["pcb", "magnetic_tile"])
    return calls


def test_the_names_are_read_off_what_the_image_ships():
    """Not a list in this module: a use-case is added by adding its script, and the error message
    that lists them has to be right without anyone remembering to update it."""
    assert "pcb" in usecase.names()
    assert all("prepare_" not in n and "_defect" not in n for n in usecase.names())


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        ("usecase://", "names no use-case"),
        ("usecase://nope", "not a use-case this image ships"),
    ],
)
def test_a_name_this_image_cannot_run_is_refused_with_the_ones_it_can(uri, match):
    with pytest.raises(ValueError, match=match):
        usecase.parse(uri)
    with pytest.raises(ValueError, match="Available|Write"):
        usecase.parse(uri)


def test_a_use_case_is_run_as_a_subprocess_of_this_interpreter(monkeypatch, tmp_path):
    """Not imported. These scripts report failure with sys.exit, which an import would raise through
    the middle of a fetch, and they download while printing -- so a cancelled run has to be able to
    stop them, which is what childproc.stream is for."""
    calls = _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    work = tmp_path / "work"

    usecase.prepare("usecase://pcb", work)

    (argv,) = calls
    assert argv[:4] == [sys.executable, "-u", "-m", "anomalygen.scripts.datasets.prepare_pcb_defect"]
    assert argv[argv.index("--output_dir") + 1] == str(work)


def test_the_child_is_unbuffered_so_its_phases_arrive_while_they_happen(monkeypatch, tmp_path):
    """The child's stdout is a pipe, so without -u the phase lines it prints sit in the buffer until
    exit. Measured on a print / sleep / stderr / exit program: without -u every line arrived at the
    end AND the error came out ahead of the phases that led to it -- under a RuntimeError that says
    "its own output above says why". A download that takes minutes also logged nothing while it ran,
    and a SIGTERM lost the lines entirely."""
    calls = _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])

    usecase.prepare("usecase://pcb", tmp_path / "work")

    (argv,) = calls
    assert "-u" in argv[: argv.index("-m")], "-u must be an interpreter flag, before -m"


def test_a_script_that_fails_fails_the_fetch(monkeypatch, tmp_path):
    _fake_script(monkeypatch, tmp_path, code=3)
    with pytest.raises(RuntimeError, match="exited 3"):
        usecase.prepare("usecase://pcb", tmp_path / "work")


def test_a_script_that_writes_nothing_is_a_failure_even_at_exit_zero(monkeypatch, tmp_path):
    """The one outcome that would otherwise pass: fetch would install an empty tree over the
    destination and the run would fail later, at a stage that cannot say why."""
    _fake_script(monkeypatch, tmp_path, writes=[], code=0)
    with pytest.raises(RuntimeError, match="exited 0 but wrote no files"):
        usecase.prepare("usecase://pcb", tmp_path / "work")


def test_a_prepared_use_case_reaches_the_destination_with_its_tree_intact(monkeypatch, tmp_path):
    """The whole point of materialising rather than adding a backend: once the script has run, this
    is the local-directory path, so the merge and the relative placement are the ones already
    tested above."""
    from anomalygen import staging

    _fake_script(
        monkeypatch,
        tmp_path,
        writes=["defect_spec.jsonl", "metal_surface/mask/MT_Crack/a_mask.png", "metal_surface/clean_image/c.jpg"],
    )
    dest = tmp_path / "dataset"

    assert staging.fetch(["usecase://pcb"], dest) == 3
    assert sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()) == [
        staging.FETCH_MARKER,
        "defect_spec.jsonl",
        "metal_surface/clean_image/c.jpg",
        "metal_surface/mask/MT_Crack/a_mask.png",
    ]


def test_the_working_copy_does_not_outlive_the_fetch(monkeypatch, tmp_path):
    """It is the size of the dataset, and on the volume the destination lives on."""
    from anomalygen import staging

    _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    dest = tmp_path / "dataset"
    staging.fetch(["usecase://pcb"], dest)

    assert not [p for p in dest.parent.iterdir() if p.name.startswith(".dataset-usecase-")]


def test_a_use_case_overlays_with_ordinary_sources(monkeypatch, tmp_path):
    """A caller correcting one file of a shipped dataset should not have to fork the script."""
    from anomalygen import staging

    _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl", "metal_surface/clean_image/c.jpg"])
    mine = tmp_path / "mine" / "metal_surface" / "clean_image"
    mine.mkdir(parents=True)
    (mine / "c.jpg").write_text("mine")

    dest = tmp_path / "dataset"
    staging.fetch(["usecase://pcb", str(tmp_path / "mine")], dest)

    assert (dest / "metal_surface" / "clean_image" / "c.jpg").read_text() == "mine"


def test_the_documented_names_are_the_ones_this_image_ships():
    """The config doc names them in prose, for a reader who cannot run `names()`.

    That makes it a second copy of a list read off the disk, and the doc is what the workflow side
    generates configs from -- so a use-case added without touching it would be invisible to exactly
    the reader who needs it, and one removed would be offered to them.
    """
    doc = pathlib.Path(__file__).resolve().parents[3] / "docs" / "pipeline" / "config.md"
    section = doc.read_text().split("### `usecase://`", 1)[1].split("###", 1)[0]
    for name in usecase.names():
        assert f"`{name}`" in section, f"{name} is shipped but not documented in {doc.name}"


def test_check_source_accepts_a_name_and_refuses_a_typo():
    """_check_fetch_spec runs this before any stage, so a typo costs nothing."""
    store.check_source("usecase://pcb")
    with pytest.raises(ValueError, match="not a use-case this image ships"):
        store.check_source("usecase://pcbb")


# ── overriding one path inside a fetched dataset ──────────────────────────────
#
# Local sources throughout: what has to be pinned here is what an override DOES to the merged tree,
# and a local directory reaches the same merge as a remote prefix through the same code.


def _dataset(root, files):
    """A dataset directory holding ``files`` ({relative path: body}), returned as a source string."""
    for rel, body in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return str(root)


def test_an_override_replaces_the_path_it_names(tmp_path):
    """The point of "generate from these three boards" is that the dataset's own are NOT used, and a
    pool of two hundred with three added is still a pool of two hundred and three."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {f"IC/clean_image/{n}.jpg": n for n in "abcd"})
    demo = _dataset(tmp_path / "demo", {"x.jpg": "x", "y.jpg": "y"})

    dest = tmp_path / "input"
    staging.fetch([source], dest, override={"IC/clean_image": [demo]})

    assert sorted(p.name for p in (dest / "IC" / "clean_image").iterdir()) == ["x.jpg", "y.jpg"]


def test_an_override_leaves_the_rest_of_the_dataset_alone(tmp_path):
    """It names one path. The submask bank and the spec beside it are the reason to fetch a dataset
    at all, and replacing the clean images must not cost them."""
    from anomalygen import staging

    source = _dataset(
        tmp_path / "ds",
        {
            "IC/clean_image/a.jpg": "a",
            "IC/cad_mask/a.png": "cad",
            "IC/mask/bridge/m.png": "m",
            "defect_spec.jsonl": "{}",
        },
    )
    demo = _dataset(tmp_path / "demo", {"x.jpg": "x"})

    dest = tmp_path / "input"
    staging.fetch([source], dest, override={"IC/clean_image": [demo]})

    assert (dest / "IC" / "cad_mask" / "a.png").read_text() == "cad"
    assert (dest / "IC" / "mask" / "bridge" / "m.png").read_text() == "m"
    assert (dest / "defect_spec.jsonl").read_text() == "{}"


def test_an_override_can_name_a_single_file(tmp_path):
    """`defect_spec.jsonl` is one file, and swapping it is how a run generates a different set of
    defect types from the same submask bank."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"defect_spec.jsonl": "shipped", "IC/clean_image/a.jpg": "a"})
    mine = tmp_path / "mine.jsonl"
    mine.write_text("mine")

    dest = tmp_path / "input"
    staging.fetch([source], dest, override={"defect_spec.jsonl": [str(mine)]})

    assert (dest / "defect_spec.jsonl").read_text() == "mine"


def test_several_sources_on_one_override_merge_in_order(tmp_path):
    """The same rule as the destination's own sources: a caller assembling one path out of a shared
    part and a per-run part must not have to stage the merge by hand."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})
    first = _dataset(tmp_path / "one", {"x.jpg": "first", "only-here.jpg": "kept"})
    second = _dataset(tmp_path / "two", {"x.jpg": "second"})

    dest = tmp_path / "input"
    staging.fetch([source], dest, override={"IC/clean_image": [first, second]})

    clean = dest / "IC" / "clean_image"
    assert (clean / "x.jpg").read_text() == "second", "later source must win"
    assert (clean / "only-here.jpg").read_text() == "kept"


def test_a_path_that_is_not_in_the_fetched_input_is_refused(tmp_path):
    """`clean_images` for `clean_image` otherwise fetches cleanly, writes an unread directory beside
    the real one, and generates from the dataset's own images as if no override had been written."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})
    demo = _dataset(tmp_path / "demo", {"x.jpg": "x"})

    with pytest.raises(RuntimeError, match="nothing to override"):
        staging.fetch([source], tmp_path / "input", override={"IC/clean_images": [demo]})


def test_a_path_that_is_not_there_is_refused_with_what_is(tmp_path):
    """A typo is found by reading the names beside it, so the message carries them."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a", "IC/cad_mask/a.png": "c"})
    demo = _dataset(tmp_path / "demo", {"x.jpg": "x"})

    with pytest.raises(RuntimeError, match="cad_mask"):
        staging.fetch([source], tmp_path / "input", override={"IC/clean_images": [demo]})


def test_a_prefix_onto_a_file_says_so(tmp_path):
    """A directory onto `defect_spec.jsonl` replaces that file with a directory of the same name, and
    the failure arrives stages later as a JSON error about something that is not a file."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a", "defect_spec.jsonl": "{}"})
    demo = _dataset(tmp_path / "demo", {"x.jpg": "x"})

    with pytest.raises(RuntimeError, match="name a prefix"):
        staging.fetch([source], tmp_path / "input", override={"defect_spec.jsonl": [demo]})


def test_a_directory_can_be_overridden_with_one_named_file(tmp_path):
    """ "Generate from THIS board" is the common request, and the board is one file in a repo of
    several -- there is no prefix that contains only it, short of re-uploading one."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {f"IC/clean_image/{n}.jpg": n for n in "abcd"})
    chosen = tmp_path / "0402_H060.jpg"
    chosen.write_text("the demo board")

    dest = tmp_path / "input"
    staging.fetch([source], dest, override={"IC/clean_image": [str(chosen)]})

    clean = dest / "IC" / "clean_image"
    assert [p.name for p in clean.iterdir()] == ["0402_H060.jpg"]
    assert (clean / "0402_H060.jpg").read_text() == "the demo board"


def test_named_files_keep_their_own_names(tmp_path):
    """The key names the directory, not the file. A cad texture pairs clean image to cad_mask by
    stem, so renaming one under the hood would break the pairing with nothing to point at."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a", "IC/cad_mask/a.png": "a"})
    for name in ("0402_H060.jpg", "0603_H100.jpg"):
        (tmp_path / name).write_text(name)
    mask = tmp_path / "0402_H060.png"
    mask.write_text("mask")

    dest = tmp_path / "input"
    staging.fetch(
        [source],
        dest,
        override={
            "IC/clean_image": [str(tmp_path / "0402_H060.jpg"), str(tmp_path / "0603_H100.jpg")],
            "IC/cad_mask": [str(mask)],
        },
    )

    assert sorted(p.name for p in (dest / "IC" / "clean_image").iterdir()) == ["0402_H060.jpg", "0603_H100.jpg"]
    assert [p.name for p in (dest / "IC" / "cad_mask").iterdir()] == ["0402_H060.png"]


def test_named_files_and_a_prefix_fill_one_directory_together(tmp_path):
    """Same rule as everywhere else in fetch: sources merge in order, later winning."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})
    pool = _dataset(tmp_path / "pool", {"x.jpg": "from the pool", "keep.jpg": "kept"})
    named = tmp_path / "x.jpg"
    named.write_text("named, and later")

    dest = tmp_path / "input"
    staging.fetch([source], dest, override={"IC/clean_image": [pool, str(named)]})

    clean = dest / "IC" / "clean_image"
    assert sorted(p.name for p in clean.iterdir()) == ["keep.jpg", "x.jpg"]
    assert (clean / "x.jpg").read_text() == "named, and later"


def test_a_remote_file_keeps_the_name_the_url_ends_in(tmp_path, monkeypatch):
    """The demo images are in a Hugging Face repo, not on the node, so the named-file case has to
    work for a URL -- where there is no `is_file()` and the name is the last segment of the source."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})
    monkeypatch.setattr(staging, "is_object", lambda uri: True)
    monkeypatch.setattr(staging, "download", lambda uri, local: local.write_text(uri))

    dest = tmp_path / "input"
    staging.fetch([source], dest, override={"IC/clean_image": ["s3://b//0402_H060.jpg"]})

    assert [p.name for p in (dest / "IC" / "clean_image").iterdir()] == ["0402_H060.jpg"]


@pytest.mark.parametrize("rel", ["/etc/cron.d/job", "../outside", "IC/../../outside", ".", "./"])
def test_an_override_cannot_name_a_path_outside_the_destination(tmp_path, rel):
    """`Path(dest) / "/etc/..."` IS the absolute path, and an override DELETES what it names."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("not this run's")

    with pytest.raises(ValueError, match="INSIDE the destination"):
        staging.fetch([source], tmp_path / "input", override={rel: [str(tmp_path / "ds")]})
    assert (outside / "keep.txt").exists()


def test_a_use_case_is_not_something_an_override_can_name(tmp_path, monkeypatch):
    """A prepare script writes textures and a defect_spec -- a whole dataset, not one path in one."""
    from anomalygen import staging

    _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})

    with pytest.raises(ValueError, match="prepares a whole dataset"):
        staging.fetch([source], tmp_path / "input", override={"IC/clean_image": ["usecase://pcb"]})


def test_an_override_on_a_single_file_destination_is_refused(tmp_path, monkeypatch):
    """`checkpoint:` is one file. An override addresses paths inside a tree, and there is no tree."""
    from anomalygen import staging

    given = tmp_path / "m.pt"
    given.write_text("weights")

    with pytest.raises(ValueError, match="nowhere to go"):
        staging.fetch([str(given)], tmp_path / "ckpt.pt", override={"IC/clean_image": [str(tmp_path / "m.pt")]})


def test_an_override_that_fails_leaves_the_previous_dataset_alone(tmp_path):
    """Applied inside the staging directory, before the swap. After it, the destination would spend a
    moment holding the dataset the config asked NOT to generate from -- and a run cancelled in that
    moment leaves exactly that on disk, looking complete."""
    from anomalygen import staging

    dest = tmp_path / "input"
    (dest / "IC" / "clean_image").mkdir(parents=True)
    (dest / "IC" / "clean_image" / "previous.jpg").write_text("previous")
    (dest / staging.FETCH_MARKER).touch()  # a re-run over what a previous fetch wrote

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RuntimeError, match="nothing to fetch under"):
        staging.fetch([source], dest, override={"IC/clean_image": [str(empty)]})
    assert (dest / "IC" / "clean_image" / "previous.jpg").read_text() == "previous"
    assert not (dest / "IC" / "clean_image" / "a.jpg").exists()


def test_no_override_is_the_fetch_that_was_there_before(tmp_path):
    """The argument is optional, and every existing caller passes nothing."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})
    dest = tmp_path / "input"
    assert staging.fetch([source], dest) == 1
    assert (dest / "IC" / "clean_image" / "a.jpg").read_text() == "a"


# ── use-case parameters ───────────────────────────────────────────────────────


def test_a_parameter_with_a_value_is_a_file_and_a_bare_one_is_a_flag():
    """The two forms are kept apart on purpose: guessing from the value whether it is a path to
    fetch or a literal to pass through is a guess that is wrong silently."""
    spec = usecase.parse("usecase://phone_screen?zip=s3://b/mobile-screen.zip&masks_only")
    assert spec.name == "phone_screen"
    assert spec.files == {"zip": "s3://b/mobile-screen.zip"}
    assert spec.flags == ("masks_only",)


def test_a_use_case_without_parameters_still_parses():
    spec = usecase.parse("usecase://pcb")
    assert (spec.name, spec.files, spec.flags) == ("pcb", {}, ())


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        ("usecase://pcb?output_dir=/etc", "must not be 'output_dir'"),
        # argparse takes any unambiguous abbreviation of a long option, so these reached the child as
        # --output/-o and were read as --output_dir: the dataset went where the config said, and the
        # run failed with "wrote no files" somewhere else entirely.
        ("usecase://pcb?output=/etc", "must not be 'output_dir'"),
        ("usecase://pcb?o=/etc", "must not be 'output_dir'"),
        ("usecase://pcb?--rm=x", "not a usable parameter name"),
        ("usecase://pcb?Zip=x", "not a usable parameter name"),
        ("usecase://pcb?zip=a&zip=b", "given twice"),
        ("usecase://pcb?zip=", "has no value"),
        ("usecase://pcb?zip=usecase://pcb", "cannot be another use-case"),
    ],
)
def test_a_parameter_that_could_not_be_passed_safely_is_refused(uri, match):
    """A parameter becomes a command-line flag, and a config is a GENERATED artifact in this
    pipeline -- so a value that lands in an argv is checked rather than passed on."""
    with pytest.raises(ValueError, match=match):
        usecase.parse(uri)


def test_the_script_is_called_with_the_parameters_the_uri_named(monkeypatch, tmp_path):
    from anomalygen import staging

    calls = _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    monkeypatch.setattr(usecase, "names", lambda: ["phone_screen"])
    zip_path = tmp_path / "mobile-screen.zip"
    zip_path.write_text("roboflow")

    # Read while the script is running: the working copy is deleted when the fetch ends, which is
    # the point of it, so afterwards there is nothing left to assert against.
    seen = {}
    inner = usecase.childproc.stream

    def run(cmd, on_line, **kw):
        argv = [str(c) for c in cmd]
        seen["zip"] = pathlib.Path(argv[argv.index("--zip") + 1]).read_text()
        return inner(cmd, on_line, **kw)

    monkeypatch.setattr(usecase.childproc, "stream", run)
    staging.fetch([f"usecase://phone_screen?zip={zip_path}&masks_only"], tmp_path / "ds")

    argv = calls[0]
    assert argv[argv.index("--zip") + 1].endswith("zip-mobile-screen.zip")
    assert "--masks_only" in argv
    assert seen["zip"] == "roboflow", "the script must be handed the file the parameter named"


def test_a_parameter_file_is_not_left_in_the_prepared_dataset(monkeypatch, tmp_path):
    """The prepared directory becomes a fetch SOURCE, so a zip left inside it would be copied into
    the input tree as if the script had produced it."""
    from anomalygen import staging

    _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    monkeypatch.setattr(usecase, "names", lambda: ["phone_screen"])
    zip_path = tmp_path / "mobile-screen.zip"
    zip_path.write_text("roboflow")

    dest = tmp_path / "ds"
    staging.fetch([f"usecase://phone_screen?zip={zip_path}"], dest)

    landed = [p.name for p in dest.rglob("*") if p.is_file()]
    assert not [n for n in landed if "mobile-screen.zip" in n], f"the zip was copied into the dataset: {landed}"
    assert not [p for p in dest.parent.iterdir() if p.name.startswith(".ds-usecase-")]


def test_a_parameter_naming_a_directory_is_refused(monkeypatch, tmp_path):
    """The script is handed a path to a file, not a directory to search."""
    from anomalygen import staging

    _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    monkeypatch.setattr(usecase, "names", lambda: ["phone_screen"])
    somewhere = tmp_path / "zips"
    somewhere.mkdir()
    (somewhere / "a.zip").write_text("x")

    with pytest.raises(ValueError, match="prefix or a directory"):
        staging.fetch([f"usecase://phone_screen?zip={somewhere}"], tmp_path / "ds")


def test_a_parameter_source_is_checked_where_every_other_source_is(tmp_path):
    """At config load, rather than after the script has been started and is waiting for a file that
    was never going to arrive."""
    with pytest.raises(ValueError, match="parameter 'zip'"):
        store.check_source(f"usecase://phone_screen?zip={tmp_path / 'absent.zip'}")


def test_an_unresolved_parameter_never_reaches_the_command_line(monkeypatch, tmp_path):
    """A missing key would otherwise reach the script as a flag with no argument, which argparse
    reports as a usage error about a flag nobody wrote."""
    _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    monkeypatch.setattr(usecase, "names", lambda: ["phone_screen"])
    with pytest.raises(RuntimeError, match="were not all resolved"):
        usecase.prepare("usecase://phone_screen?zip=s3://b/x.zip", tmp_path / "ds", files={})


def test_a_destination_this_run_may_not_replace_is_refused_before_the_script_runs(monkeypatch, tmp_path):
    """`input: /data/mine` plus a use-case fetched the whole gated dataset and THEN refused to
    replace the directory -- minutes of bytes moved for an outcome that was decided before the first
    one. A use-case fills a directory, so the guard needs nothing listed to answer."""
    from anomalygen import staging

    calls = _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    theirs = tmp_path / "mine"
    (theirs / "T").mkdir(parents=True)
    (theirs / "T" / "j1.png").write_text("five thousand of these")

    with pytest.raises(RuntimeError, match="no previous fetch wrote it"):
        staging.fetch(["usecase://pcb"], theirs)

    assert calls == [], "the script must not have been run"
    assert (theirs / "T" / "j1.png").read_text() == "five thousand of these"


def test_an_override_source_that_holds_nothing_is_found_before_the_dataset_is_fetched(monkeypatch, tmp_path):
    """check_override reads the shape and the scheme only, so a prefix an upstream task has not
    written yet used to be reported after a `usecase://` download of several minutes."""
    from anomalygen import staging

    calls = _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    empty = tmp_path / "not-written-yet"
    empty.mkdir()

    with pytest.raises(RuntimeError, match="nothing to fetch under"):
        staging.fetch(["usecase://pcb"], tmp_path / "ds", override={"IC/clean_image": [str(empty)]})

    assert calls == [], "the use-case must not have been prepared"


def test_an_override_source_is_listed_before_the_merge_is_copied(monkeypatch, tmp_path):
    """The same pass, for an ordinary source: nothing of the dataset is copied for a run that the
    override is going to refuse."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a"})
    empty = tmp_path / "empty"
    empty.mkdir()
    copied = []
    monkeypatch.setattr(staging, "_copy_one", lambda src, local: copied.append(src))

    with pytest.raises(RuntimeError, match="nothing to fetch under"):
        staging.fetch([source], tmp_path / "input", override={"IC/clean_image": [str(empty)]})

    assert copied == [], "the dataset must not have been copied"


def test_the_scripts_working_cache_does_not_become_dataset_content(monkeypatch, tmp_path):
    """Both HF scripts download with local_dir=, which keeps per-file metadata in .cache/huggingface/,
    and the magnetic-tile script keeps the raw GitHub archive there: 58 MB of the 61 MB it produces,
    copied twice on the way into `input`."""
    from anomalygen import staging

    _fake_script(
        monkeypatch,
        tmp_path,
        writes=["defect_spec.jsonl", "T/clean_image/a.jpg", ".cache/huggingface/blob", ".cache/archive.zip"],
    )

    dest = tmp_path / "ds"
    staging.fetch(["usecase://pcb"], dest)

    assert not (dest / ".cache").exists()
    assert (dest / "T" / "clean_image" / "a.jpg").exists(), "the dataset itself must survive"


def test_a_cache_directory_deeper_in_the_dataset_is_left_alone(tmp_path, monkeypatch):
    """Only the top level. A dataset is free to carry a directory of that name, and dropping data
    silently is worse than carrying a few files nobody wanted."""
    from anomalygen import staging

    _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl", "T/.cache/keep.png"])

    dest = tmp_path / "ds"
    staging.fetch(["usecase://pcb"], dest)

    assert (dest / "T" / ".cache" / "keep.png").exists()


def test_a_script_that_wrote_only_working_state_is_a_failure(monkeypatch, tmp_path):
    """The count that the exit-0-wrote-nothing guard reads is taken before the drop, so without a
    second check this would install an empty dataset over the destination."""
    from anomalygen import staging

    _fake_script(monkeypatch, tmp_path, writes=[".cache/huggingface/blob"])

    with pytest.raises(RuntimeError, match="only \\['.cache'\\] working state"):
        staging.fetch(["usecase://pcb"], tmp_path / "ds")


def test_our_own_flag_is_passed_last_so_it_cannot_be_outvoted(monkeypatch, tmp_path):
    """The half that does not depend on knowing how argparse abbreviates: whatever a parameter is
    matched to, a later value for the same destination replaces an earlier one. Both halves are
    needed -- the name check knows the flag, this one does not have to."""
    calls = _fake_script(monkeypatch, tmp_path, writes=["defect_spec.jsonl"])
    monkeypatch.setattr(usecase, "names", lambda: ["phone_screen"])
    zip_path = tmp_path / "z.zip"
    zip_path.write_text("x")

    usecase.prepare(f"usecase://phone_screen?zip={zip_path}", tmp_path / "into", files={"zip": zip_path})

    (argv,) = calls
    assert argv[-2:] == ["--output_dir", str(tmp_path / "into")], f"--output_dir must be last: {argv}"


def test_an_override_key_of_dot_does_not_replace_the_whole_dataset(tmp_path):
    """`Path(".").parts` is (), so `.` is neither empty, absolute nor climbing: it resolved to the
    staging directory itself, and "replace one path inside the dataset" became "rmtree the dataset
    and refill it from this key" -- after the use-case script had run for nothing."""
    from anomalygen import staging

    source = _dataset(tmp_path / "ds", {"IC/clean_image/a.jpg": "a", "defect_spec.jsonl": "{}"})
    mine = _dataset(tmp_path / "mine", {"x.jpg": "x"})

    with pytest.raises(ValueError, match="INSIDE the destination"):
        staging.fetch([source], tmp_path / "input", override={".": [mine]})
