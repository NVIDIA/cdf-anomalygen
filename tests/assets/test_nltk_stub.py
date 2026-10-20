# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``assets/nltk_stub``, the stand-in for the parts of nltk the guardrail blocklist uses.

The stub is loaded from its source file under its own module name, so these tests run the same
whether or not the real nltk is installed. Most use a small synthetic ``nltk_data`` tree; the last
group runs against the real Punkt and WordNet data, with expectations recorded from nltk
3.10.3, and is skipped where that data is not downloaded.
"""

import importlib.util
import pathlib
import zipfile

import pytest

from anomalygen.data.utils import caption_for_anomaly_type

_REPO = pathlib.Path(__file__).resolve().parents[2]
_STUB = _REPO / "assets" / "nltk_stub" / "nltk" / "__init__.py"
# The guardrail checkpoint's nltk_data, else the same Punkt and WordNet files from NLTK's data
# repository, which CI fetches to checkpoints/nltk_data because the checkpoint is gated.
_CHECKPOINT_DATA = sorted(
    (_REPO / "checkpoints" / "hf" / "models--nvidia--Cosmos-Guardrail1" / "snapshots").glob("*/blocklist/nltk_data")
) + [p for p in [_REPO / "checkpoints" / "nltk_data"] if p.is_dir()]


@pytest.fixture
def stub():
    """A fresh copy of the stub module, so one test's ``data.path`` never leaks into another."""
    spec = importlib.util.spec_from_file_location("nltk_stub_under_test", _STUB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INDEX_NOUN = "  header line, skipped like WordNet's licence block\nbus n 1\ncable n 1\nchurch n 1\ncrack n 1\ndog n 1\ngoose n 1\n"
_NOUN_EXC = "geese goose\n"


def _wordnet_dir(root: pathlib.Path) -> None:
    wordnet = root / "corpora" / "wordnet"
    wordnet.mkdir(parents=True)
    (wordnet / "index.noun").write_text(_INDEX_NOUN)
    (wordnet / "noun.exc").write_text(_NOUN_EXC)


def _punkt_dir(root: pathlib.Path) -> None:
    english = root / "tokenizers" / "punkt_tab" / "english"
    english.mkdir(parents=True)
    (english / "abbrev_types.txt").write_text("dr\ninc\n")
    (english / "collocations.tab").write_text("")
    (english / "sent_starters.txt").write_text("it\n")
    (english / "ortho_context.tab").write_text("")


@pytest.fixture
def nltk_data(tmp_path):
    _wordnet_dir(tmp_path)
    _punkt_dir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "word, lemma",
    [
        ("dogs", "dog"),  # plain "s"
        ("churches", "church"),  # "ches" -> "ch"
        ("geese", "goose"),  # exception list
        ("bus", "bus"),  # the word itself is a lemma, and the shortest
        ("cracks.", "cracks."),  # not in WordNet in any form: unchanged
        ("Dogs", "Dogs"),  # WordNet keys are lower case, as in nltk
    ],
)
def test_lemmatize_nouns(stub, nltk_data, word, lemma):
    stub.data.path.append(str(nltk_data))
    assert stub.WordNetLemmatizer().lemmatize(word) == lemma


def test_lemmatize_reads_wordnet_from_its_zip(stub, tmp_path):
    (tmp_path / "corpora").mkdir()
    with zipfile.ZipFile(tmp_path / "corpora" / "wordnet.zip", "w") as archive:
        archive.writestr("wordnet/index.noun", _INDEX_NOUN)
        archive.writestr("wordnet/noun.exc", _NOUN_EXC)
    stub.data.path.append(str(tmp_path))
    assert stub.WordNetLemmatizer().lemmatize("geese") == "goose"


def test_lemmatize_rejects_other_parts_of_speech(stub, nltk_data):
    stub.data.path.append(str(nltk_data))
    with pytest.raises(NotImplementedError, match="nouns only"):
        stub.WordNetLemmatizer().lemmatize("running", pos="v")


def test_missing_data_names_the_search_path(stub, tmp_path):
    stub.data.path.append(str(tmp_path))
    with pytest.raises(LookupError, match="corpora/wordnet not found"):
        stub.WordNetLemmatizer().lemmatize("dogs")


def test_word_tokenize_splits_sentences_but_keeps_abbreviations(stub, nltk_data):
    stub.data.path.append(str(nltk_data))
    assert stub.sent_tokenize("The board was made by Acme Inc. in 2020. It has two cracks.") == [
        "The board was made by Acme Inc. in 2020.",
        "It has two cracks.",
    ]
    assert stub.word_tokenize("Dr. Smith's boards can't pass. It isn't clean!") == [
        "Dr.", "Smith", "'s", "boards", "ca", "n't", "pass", ".", "It", "is", "n't", "clean", "!",
    ]  # fmt: skip


def test_rejects_other_languages(stub, nltk_data):
    stub.data.path.append(str(nltk_data))
    with pytest.raises(NotImplementedError, match="English only"):
        stub.word_tokenize("Ein Satz.", language="german")


def test_anything_else_fails_loudly(stub):
    with pytest.raises(AttributeError, match="nltk_stub"):
        stub.pos_tag  # noqa: B018


# Recorded from nltk 3.10.3 on the Cosmos-Guardrail1 data: what the blocklist matches against after
# lemmatizing a caption (word_tokenize, lemmatize each token, join with spaces).
_TAIL = (
    " . Defective , Non-conforming , Substandard , Irregular , Damaged , Misaligned , Incomplete , Malformed ,"
    " Contaminated , Under-processed , Faulty , Blemished , Expired ."
)


@pytest.mark.skipif(not _CHECKPOINT_DATA, reason="requires the guardrail checkpoint or checkpoints/nltk_data")
@pytest.mark.parametrize(
    "anomaly_type, lemmatized",
    [
        ("metal_surface+MT_Crack", "Add MT Crack to a close-up look of a metal surface" + _TAIL),
        ("IC+bridges", "Add bridge to a close-up look of a IC" + _TAIL),
        ("wafer.v2+scratches", "Add scratch to a close-up look of a wafer.v2" + _TAIL),
        ("cables+cut_outer_insulations", "Add cut outer insulation to a close-up look of a cable" + _TAIL),
        ("etc+cracks", "Add crack to a close-up look of a etc" + _TAIL),  # a Punkt abbreviation before the period
        ("x+cracks", "Add crack to a close-up look of a x" + _TAIL),  # an initial
        ("3+cracks", "Add crack to a close-up look of a 3" + _TAIL),  # a number
    ],
)
def test_matches_nltk_on_guardrail_captions(stub, anomaly_type, lemmatized):
    stub.data.path.append(str(_CHECKPOINT_DATA[0]))
    lemmatizer = stub.WordNetLemmatizer()
    tokens = stub.word_tokenize(caption_for_anomaly_type(anomaly_type))
    assert " ".join(lemmatizer.lemmatize(t) for t in tokens) == lemmatized
