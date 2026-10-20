# pycountry stub

An **import-only stub** for the `pycountry` package. It exposes the five ISO databases
`pydantic_extra_types` reads (`countries`, `currencies`, `languages`, `language_families`,
`scripts`) as empty collections — enough to satisfy imports, nothing functional.

## Why it exists

vllm imports its structured-output backends when the API server loads, one of which imports
mistral-common's tokenizer support, which imports `mistral_common.protocol.transcription.request`,
which imports `pydantic_extra_types.language_code`, which does `import pycountry` at the top and
raises without it. So `vllm serve` cannot start without `pycountry` — for any model, gemma
included — even though only audio transcription requests ever validate a language code, and this
image serves vision models only.

The real `pycountry` is LGPL-2.1-only. The stub keeps it out of the shipped image without
touching vllm.

## How it's wired

`docker/Dockerfile` and `scripts/env_setup.sh` uninstall `pycountry` right after vllm and install
this package in its place with `--no-deps`; in the Dockerfile that happens in the shared stubs layer,
which also checks that vllm's API server imports. The licence generator and the OSS bake treat
`pycountry-stub` as this repo's own code.
