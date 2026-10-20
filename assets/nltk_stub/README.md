# nltk stub

A stand-in for the `nltk` package that implements exactly the three things cosmos-framework's
guardrail `Blocklist` (`cosmos_framework/auxiliary/guardrail/blocklist/blocklist.py`) uses:

- `nltk.data.path`, the list of data directories the blocklist appends its checkpoint's
  `nltk_data` to;
- `nltk.word_tokenize(text)`, English Punkt sentence splitting followed by NLTK's Treebank word
  rules;
- `nltk.WordNetLemmatizer().lemmatize(word)`, WordNet noun lemmatization (`pos="n"`, the default).

Anything else raises `AttributeError` naming the stub, so a framework change that starts using
more of nltk fails loudly instead of behaving differently.

## Why it exists

The real `nltk` carries CVE-2026-81726 (GHSA-8mgp-746c-j5xp, file-sandbox bypass in its model
import/export APIs) with no fixed release on PyPI. The pipeline never calls the affected APIs, but
the package is flagged as long as it ships. The blocklist needs only tokenization and noun
lemmatization, and the Cosmos-Guardrail1 checkpoint already ships the data for both
(`blocklist/nltk_data/tokenizers/punkt_tab/english/` and `corpora/wordnet.zip`).

The stub reads that data with plain file I/O, so it also needs none of the workarounds the real
package's path-security checks require for files inside the Hugging Face cache.

## Behaviour

The algorithms are ported from NLTK 3.10.3 (Apache License 2.0): Punkt's sentence-boundary
detection with the shipped English parameters, `NLTKWordTokenizer`'s regular expressions, and
WordNet's noun `morphy`. Against the real package, over every guardrail caption built from each
WordNet noun, its plural forms, every Punkt abbreviation, single letters, numbers and the
blocklist's own terms (in both the texture and the defect slot), the token lists and lemmatized
strings are identical, and the framework's `Blocklist` returns the same verdicts.

Only English and nouns are provided; other languages or parts of speech raise
`NotImplementedError`.

## How it's wired

`requirements.txt` no longer lists `nltk`, and nothing else in the image depends on it.
`docker/Dockerfile` (in its shared stubs layer), `scripts/env_setup.sh` and
`scripts/env_setup_ci_cpu.sh` install this package with `--no-deps`; the Dockerfile also fails the
build if a resolve ever brings the real `nltk` back.

An environment updated in place can still import the real package, because `uv pip install -r`
does not remove it. The guardrail works with either package. The stub sets `nltk.IS_STUB`, and
`create_guardrail_runners` applies the real package's workarounds only when that marker is
absent: it adds the checkpoint to nltk's allowlist and reads a copy of `nltk_data` without
symlinks. To remove the real package, rebuild the environment, or run `uv pip uninstall nltk`
and install the stub again.
Because the module is ported NLTK code, the package carries NLTK's `LICENSE.txt` and is attributed
in `LICENSE-3rd-party.txt` as `nltk-stub`. The OSS bake treats it as this repo's own source, which
the image ships under `assets/nltk_stub/`.
