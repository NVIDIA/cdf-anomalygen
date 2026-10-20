# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (C) 2001-2026 NLTK Project
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for ``nltk``: the tokenizer and noun lemmatizer cosmos-framework's guardrail blocklist uses.

Provides ``data.path``, ``word_tokenize`` and ``WordNetLemmatizer``, reading the Punkt and WordNet
data the guardrail checkpoint ships under ``nltk_data`` with plain file I/O. The algorithms are
ported from NLTK 3.10.3 (``nltk/tokenize/punkt.py``, ``nltk/tokenize/destructive.py``,
``nltk/corpus/reader/wordnet.py``) and produce the same tokens and lemmas. Any other attribute
raises, so a caller that needs more of nltk fails at the call rather than behaving differently.
"""

from __future__ import annotations

import os
import re
import string
import zipfile
from functools import lru_cache

__all__ = ["data", "word_tokenize", "sent_tokenize", "WordNetLemmatizer"]

# Tells this stand-in apart from the real package, which has the same import name.
IS_STUB = True


class _Data:
    """``nltk.data``: only the search path, which callers extend with their own data directories."""

    def __init__(self) -> None:
        self.path: list[str] = []


data = _Data()


def __getattr__(name: str):
    raise AttributeError(f"module 'nltk' (assets/nltk_stub) has no attribute {name!r}; the stub provides {__all__}")


def _reader(rel: str):
    """``read(member)`` for ``rel`` (e.g. ``corpora/wordnet``) on ``data.path``, a directory or its ``.zip``."""
    for root in data.path:
        base = os.path.join(root, rel)
        if os.path.isdir(base):

            def read(member: str, base: str = base) -> str:
                with open(os.path.join(base, member), encoding="utf-8") as f:
                    return f.read()

            return read
        if os.path.isfile(base + ".zip"):
            archive = zipfile.ZipFile(base + ".zip")
            prefix = os.path.basename(rel)
            return lambda member: archive.read(f"{prefix}/{member}").decode("utf-8")
    raise LookupError(f"{rel} not found on nltk.data.path={data.path}")


# --- WordNet noun lemmatizer (nltk.corpus.reader.wordnet._morphy, pos="n") ---------------------------

_NOUN_SUBSTITUTIONS = (
    ("s", ""),
    ("ses", "s"),
    ("ves", "f"),
    ("xes", "x"),
    ("zes", "z"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("men", "man"),
    ("ies", "y"),
)


@lru_cache(maxsize=None)
def _wordnet_nouns(search_path: tuple[str, ...]) -> tuple[frozenset[str], dict[str, list[str]]]:
    read = _reader("corpora/wordnet")
    lemmas = frozenset(
        line.split(" ", 1)[0] for line in read("index.noun").splitlines() if line and not line.startswith(" ")
    )
    exceptions = {}
    for line in read("noun.exc").splitlines():
        terms = line.split()
        if terms:
            exceptions[terms[0]] = terms[1:]
    return lemmas, exceptions


class WordNetLemmatizer:
    """``nltk.WordNetLemmatizer``: the shortest WordNet lemma of a noun, or the word unchanged."""

    def lemmatize(self, word: str, pos: str = "n") -> str:
        if pos != "n":
            raise NotImplementedError(f"the nltk stub lemmatizes nouns only, not pos={pos!r}")
        lemmas, exceptions = _wordnet_nouns(tuple(data.path))
        if word in exceptions:
            forms = exceptions[word]
        else:
            forms = [word[: -len(old)] + new for old, new in _NOUN_SUBSTITUTIONS if word.endswith(old)]
        found = list(dict.fromkeys(f for f in [word, *forms] if f in lemmas))
        return min(found, key=len) if found else word


# --- Treebank word rules (nltk.tokenize.destructive.NLTKWordTokenizer) -------------------------------

_PAD = r" \g<0> "  # surround the whole match with spaces, so it splits off as a token

_STARTING_QUOTES = [
    (re.compile("([«“‘„]|[`]+)", re.U), r" \1 "),
    (re.compile(r"^\""), r"``"),
    (re.compile(r"(``)"), r" \1 "),
    (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
    (re.compile(r"(?i)(?<!\w)(\')(?!(?:re|ve|ll|m|t|s|d|n)\b)(?=\w)", re.U), r"\1 "),
]
_ENDING_QUOTES = [
    (re.compile("([»”’])", re.U), r" \1 "),
    (re.compile(r"''"), " '' "),
    (re.compile(r'"'), " '' "),
    (re.compile(r"\s+"), " "),
    (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
    (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
]
_PUNCTUATION = [
    (re.compile(r'([^\.])(\.)([\]\)}>"\'' "»”’ " r"]*)\s*$", re.U), r"\1 \2 \3 "),
    (re.compile(r"([:,])([^\d])"), r" \1 \2"),
    (re.compile(r"([:,])$"), r" \1 "),
    (re.compile(r"\.{2,}", re.U), _PAD),
    (re.compile(r"[;@#$%&]"), _PAD),
    (re.compile(r"[‒-―]", re.U), _PAD),
    (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r"\1 \2\3 "),
    (re.compile(r"[?!]"), _PAD),
    (re.compile(r"([^'])' "), r"\1 ' "),
    (re.compile(r"[*]", re.U), _PAD),
]
_PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), _PAD)
_DOUBLE_DASHES = (re.compile(r"--"), r" -- ")
_CONTRACTIONS = [
    re.compile(p)
    for p in (
        r"(?i)\b(can)(?#X)(not)\b",
        r"(?i)\b(d)(?#X)('ye)\b",
        r"(?i)\b(gim)(?#X)(me)\b",
        r"(?i)\b(gon)(?#X)(na)\b",
        r"(?i)\b(got)(?#X)(ta)\b",
        r"(?i)\b(lem)(?#X)(me)\b",
        r"(?i)\b(more)(?#X)('n)\b",
        r"(?i)\b(wan)(?#X)(na)(?=\s)",
        r"(?i) ('t)(?#X)(is)\b",
        r"(?i) ('t)(?#X)(was)\b",
    )
]


def _treebank_words(text: str) -> list[str]:
    for regexp, substitution in _STARTING_QUOTES + _PUNCTUATION + [_PARENS_BRACKETS, _DOUBLE_DASHES]:
        text = regexp.sub(substitution, text)
    text = f" {text} "
    for regexp, substitution in _ENDING_QUOTES:
        text = regexp.sub(substitution, text)
    for regexp in _CONTRACTIONS:
        text = regexp.sub(r" \1 \2 ", text)
    return text.split()


# --- Punkt sentence boundaries (nltk.tokenize.punkt, trained English parameters) ---------------------

_ORTHO_BEG_UC, _ORTHO_MID_UC, _ORTHO_UNK_UC = 1 << 1, 1 << 2, 1 << 3
_ORTHO_BEG_LC, _ORTHO_MID_LC, _ORTHO_UNK_LC = 1 << 4, 1 << 5, 1 << 6
_ORTHO_UC = _ORTHO_BEG_UC | _ORTHO_MID_UC | _ORTHO_UNK_UC
_ORTHO_LC = _ORTHO_BEG_LC | _ORTHO_MID_LC | _ORTHO_UNK_LC

_SENT_END_CHARS = (".", "?", "!")
_PUNCTUATION_TOKENS = (";", ":", ",", ".", "!", "?")
_NON_WORD = r"(?:[)\";}\]\*:@\'\({\[‘’“”\xab\xbb" + re.escape("?!") + r"])"
_MULTI_CHAR = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"
_WORD_START = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
_PUNKT_WORDS = re.compile(
    rf"""(
        {_MULTI_CHAR}
        |
        (?={_WORD_START})\S+?
        (?=
            \s|
            $|
            {_NON_WORD}|{_MULTI_CHAR}|
            ,(?=$|\s|{_NON_WORD}|{_MULTI_CHAR})
        )
        |
        \S
    )""",
    re.U | re.X,
)
_PERIOD_CONTEXT = re.compile(
    rf"""
        [{re.escape("".join(_SENT_END_CHARS))}]
        (?=(?P<after_tok>
            {_NON_WORD}
            |
            \s+(?P<next_tok>\S+)
        ))""",
    re.U | re.X,
)
_BOUNDARY_REALIGNMENT = re.compile(r'["\')\]}‘’“”\xab\xbb]+?(?:\s+|(?=--)|$)', re.M)
_ELLIPSIS = re.compile(r"\.\.+$")
_NUMERIC = re.compile(r"^-?[\.,]?\d[\d,\.-]*\.?$")
_INITIAL = re.compile(r"[^\W\d]\.$", re.U)


@lru_cache(maxsize=None)
def _punkt_params(search_path: tuple[str, ...]):
    read = _reader("tokenizers/punkt_tab")
    lines = lambda name: [line for line in read(f"english/{name}").splitlines() if line.strip()]  # noqa: E731
    abbrev_types = frozenset(line.strip() for line in lines("abbrev_types.txt"))
    collocations = frozenset(tuple(line.split("\t")) for line in lines("collocations.tab"))
    sent_starters = frozenset(line.strip() for line in lines("sent_starters.txt"))
    ortho_context = {key: int(value) for key, value in (line.split("\t") for line in lines("ortho_context.tab"))}
    return abbrev_types, collocations, sent_starters, ortho_context


class _Token:
    __slots__ = ("tok", "type", "period_final", "sentbreak", "abbr", "ellipsis")

    def __init__(self, tok: str) -> None:
        self.tok = tok
        self.type = _NUMERIC.sub("##number##", tok.lower())
        self.period_final = tok.endswith(".")
        self.sentbreak = self.abbr = self.ellipsis = False

    @property
    def type_no_period(self) -> str:
        return self.type[:-1] if len(self.type) > 1 and self.type[-1] == "." else self.type

    @property
    def type_no_sentperiod(self) -> str:
        return self.type_no_period if self.sentbreak else self.type


def _ortho_heuristic(token: _Token, ortho_context: dict[str, int]):
    """True if ``token`` starts a sentence, False if it does not, ``"unknown"`` otherwise."""
    if token.tok in _PUNCTUATION_TOKENS:
        return False
    context = ortho_context.get(token.type_no_sentperiod, 0)
    if token.tok[0].isupper() and (context & _ORTHO_LC) and not (context & _ORTHO_MID_UC):
        return True
    if token.tok[0].islower() and ((context & _ORTHO_UC) or not (context & _ORTHO_BEG_LC)):
        return False
    return "unknown"


def _first_pass(token: _Token, abbrev_types: frozenset[str]) -> None:
    """Mark sentence-end marks and period-final non-abbreviations as breaks, ellipses and abbreviations as such."""
    if token.tok in _SENT_END_CHARS:
        token.sentbreak = True
    elif _ELLIPSIS.match(token.tok):
        token.ellipsis = True
    elif token.period_final and not token.tok.endswith(".."):
        stem = token.tok[:-1].lower()
        if stem in abbrev_types or stem.split("-")[-1] in abbrev_types:
            token.abbr = True
        else:
            token.sentbreak = True


def _breaks_after_abbreviation(following: _Token, sent_starters: frozenset[str], ortho_context: dict[str, int]) -> bool:
    """An abbreviation or ellipsis ends a sentence when the next word reads as a sentence start."""
    if _ortho_heuristic(following, ortho_context) is True:
        return True
    return following.tok[0].isupper() and following.type_no_sentperiod in sent_starters


def _continues_after_initial_or_number(following: _Token, is_initial: bool, ortho_context: dict[str, int]) -> bool:
    """An initial or a number does not end a sentence when the next word does not start one."""
    starts_sentence = _ortho_heuristic(following, ortho_context)
    if starts_sentence is False:
        return True
    return (
        starts_sentence == "unknown"
        and is_initial
        and following.tok[0].isupper()
        and not (ortho_context.get(following.type_no_sentperiod, 0) & _ORTHO_LC)
    )


def _second_pass(token: _Token, following: _Token, params) -> None:
    """Revise a period-final token's decision from the token that follows it."""
    _, collocations, sent_starters, ortho_context = params
    typ = token.type_no_period
    is_initial = bool(_INITIAL.match(token.tok))
    if (typ, following.type_no_sentperiod) in collocations:
        token.sentbreak, token.abbr = False, True
    elif (
        (token.abbr or token.ellipsis)
        and not is_initial
        and _breaks_after_abbreviation(following, sent_starters, ortho_context)
    ):
        token.sentbreak = True
    elif (is_initial or typ == "##number##") and _continues_after_initial_or_number(
        following, is_initial, ortho_context
    ):
        token.sentbreak, token.abbr = False, True


def _annotated_tokens(text: str, params) -> list[_Token]:
    tokens = [_Token(word) for line in text.split("\n") if line.strip() for word in _PUNKT_WORDS.findall(line)]
    for token in tokens:
        _first_pass(token, params[0])
    for token, following in zip(tokens, tokens[1:]):
        if token.period_final:
            _second_pass(token, following, params)
    return tokens


def _end_contexts(text: str):
    """Each candidate boundary with the text Punkt judges it on: the word before, the mark, what follows."""
    previous_slice, previous_match = slice(0, 0), None
    for match in _PERIOD_CONTEXT.finditer(text):
        before = text[previous_slice.stop : match.start()]
        last_space = next((i for i in range(len(before) - 1, -1, -1) if before[i] in string.whitespace), 0)
        start = last_space + previous_slice.stop + 1 if last_space else previous_slice.start
        word = slice(start, match.start())
        if previous_match and previous_slice.stop <= word.start:
            yield previous_match, text[previous_slice] + previous_match.group() + previous_match.group("after_tok")
        previous_match, previous_slice = match, word
    if previous_match:
        yield previous_match, text[previous_slice] + previous_match.group() + previous_match.group("after_tok")


def _sentence_spans(text: str, params) -> list[slice]:
    """Punkt's sentence spans, before closing punctuation is moved back onto the sentence it closes."""
    spans, last_break = [], 0
    for match, context in _end_contexts(text):
        if any(token.sentbreak for token in _annotated_tokens(context, params)[:-1]):
            spans.append(slice(last_break, match.end()))
            last_break = match.start("next_tok") if match.group("next_tok") else match.end()
    spans.append(slice(last_break, len(text.rstrip())))
    return spans


def _realigned(text: str, spans: list[slice]) -> list[str]:
    """Sentences from ``spans``, with quotes or brackets that open the next span attached to this one."""
    sentences, realign = [], 0
    for span, following in zip(spans, spans[1:] + [None]):
        span = slice(span.start + realign, span.stop)
        closing = _BOUNDARY_REALIGNMENT.match(text[following]) if following is not None else None
        if closing:
            sentences.append(text[span.start : following.start + len(closing.group(0).rstrip())])
            realign = closing.end()
            continue
        realign = 0
        if text[span]:
            sentences.append(text[span])
    return sentences


def sent_tokenize(text: str, language: str = "english") -> list[str]:
    """``nltk.sent_tokenize`` with the trained English Punkt parameters."""
    if language != "english":
        raise NotImplementedError(f"the nltk stub tokenizes English only, not {language!r}")
    return _realigned(text, _sentence_spans(text, _punkt_params(tuple(data.path))))


def word_tokenize(text: str, language: str = "english", preserve_line: bool = False) -> list[str]:
    """``nltk.word_tokenize``: Punkt sentences, each split by the Treebank word rules."""
    sentences = [text] if preserve_line else sent_tokenize(text, language)
    return [token for sentence in sentences for token in _treebank_words(sentence)]
