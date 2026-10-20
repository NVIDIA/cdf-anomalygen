# The run_pipeline config

Reference for the YAML that `anomalygen.scripts.run_pipeline` takes. It states the rules rather than showing one config
and leaving them to be inferred.

`ag_config/pipeline/example.yaml` is the companion piece: a complete, commented config to copy.
This file is what you consult when deciding what a generated one must contain.

## Validate, don't guess

```shell
python -m anomalygen.scripts.run_pipeline --config <file> --dry_run
```

This is the authoritative check, and the only one this repo ships. It is the real loader and the
real preflights — every rule below, plus the ones that need the filesystem: paths exist, the defect
types named are in `defect_spec.jsonl`, the weights the selected stages will open are present,
enough GPUs are visible, the publish destination is reachable. It runs no stage. **Whatever it says
is what a real run will say**, because it is the same code.

One caveat, and it matters for a generated config: with `prepare` in `stages`, the paths
`staging.fetch` will supply do not exist yet, so they are skipped at load and checked again after
`prepare` has run. A `--dry_run` on such a config validates the config, not the inputs.

There is deliberately no JSON Schema here. One would have to restate the cross-key rules below in a
second notation, and a rule added to the loader without a matching line there would be accepted by
the schema and rejected by the run — the worst direction, because the config was "checked".
Structural validation of a generated config belongs to the layer that generates it; this document is
the specification to build it from.

## The shape of one

Three keys are always required — `input`, `output`, `recipe` — and everything else follows from
`stages`. An unknown key is an **error**, never ignored: a typo in an optional key would otherwise
leave its default silently in force and the run would look correct while doing something else.

```yaml
input:  /work/dataset      # the dataset, in the Step 1 layout
output: /work/out          # every stage writes under here
recipe: /work/recipe.yaml  # the recipe the fine-tuned checkpoint was trained with
stages: [amp, generate, export]
num_sdg: 7
checkpoint: /work/model.pt
```

## Stages, and what each one demands

`stages` may be given in any order and is reordered to the canonical one below, deduplicated. A
stage you skip must already have left its artifacts under `output` — that is how a run resumes.

| stage      | what it does                                                              | what selecting it requires                                                                             |
| ---------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `prepare`  | downloads inputs named in `staging.fetch`; provisions any missing weights | —                                                                                                      |
| `amp`      | mask placement, geometric. Writes `<output>/amp/testcase.jsonl`           | `num_sdg` (or `per_defect_counts`)                                                                     |
| `amg`      | mask placement, agentic. Writes `<output>/amg/testcase.jsonl`             | `num_sdg` (or `per_defect_counts`), and `amg_model_registry` + the two model names, **or** `amg_serve` |
| `generate` | the diffusion pass                                                        | `checkpoint`                                                                                           |
| `evaluate` | scores the batch into `kpi.json`                                          | —                                                                                                      |
| `refine`   | re-generates and re-ranks against that KPI table                          | `checkpoint`, and `evaluate` ahead of it                                                               |
| `export`   | collects the results and writes `export_manifest.json`                    | —                                                                                                      |
| `publish`  | uploads what the manifest lists                                           | `staging.publish.output`                                                                               |

**Exactly one of `amp` / `amg`.** They are two backends for one artifact; listing both is an error,
because `generate` must not have to guess which `testcase.jsonl` to read.

Default when `stages` is omitted: `[amp, generate, evaluate, refine, export]` — no `prepare`, no
`publish`, no `amg`, because each needs locations or an endpoint the config may not carry.

## Rules that span keys

These are the ones a per-key description cannot hold, and the ones a generator gets wrong. Each is
enforced at load time, before any stage runs.

- **`num_sdg` and `per_defect_counts` state the same thing.** Give either. When both are given and
  disagree, the counts win and the run says so — `num_sdg` is normalised to the sum, so the resolved
  config records what was actually generated.
- **`per_defect_counts` requires `mode: inference`.** A validation set is allocated proportionally
  over every type in `defect_spec.jsonl`, so selecting types contradicts it.
- **`output_naming: source` needs one output per clean image.** Under `amp` that means
  `amp_per_clean_pairing: on`, which in turn accepts exactly one defect type with a non-zero count.
  With `off`, the round-robin can draw two outputs from one clean image and source naming would
  collide — use `indexed`.
- **`amp_per_clean_pairing: on` is an error under `amg`.** There is no pairing step; placement is by
  model judgment, so it cannot promise one output per clean image.
- **`amp_seed` does nothing under `amg`**, for the same reason. It is accepted and ignored, with a
  note in the log.
- **`amg_serve` and `amg_model_registry` are alternatives.** `amg_serve` names a recipe this
  entrypoint starts before the stage and stops after, and it also defaults `amg_gen_model` and
  `amg_judge_model` to that recipe's model. Without it, something must already be serving and the
  registry must name it.
- **`base_checkpoint` is almost always omitted.** Left null, `generate` picks the base DCP from the
  recipe's `model_size`. Set it only to point at a base tree that is somewhere else.

`export_manifest.json` records `images` (published filename → internal generated filename) and
`input_images` (published filename → original clean-image path from the generation CSV). Both
naming modes include this association; under `indexed`, several outputs may refer to the same input.

### Seeds

`amp_seed` and `base_seed` are different axes, not a default and an override:

- `amp_seed` picks **which** base mask is drawn and where it lands — the defect's shape and position.
- `base_seed` is the diffusion noise **under** that mask — only how it is rendered.

Changing `base_seed` alone re-renders the identical mask. A caller that wants a genuinely different
image varies both.

**Vary both per task when a run emits one image.** `roi_pair` shuffles the submask bank on
`(amp_seed, defect_type)` — not the clean image, not which run this is — and a one-image run always
takes submask 0 of that shuffle, with noise index 0. Point 200 such runs at 200 crops with one seed
and all 200 share a base mask; each is plausible alone and the repetition only shows in a contact
sheet. Derive both from whatever identifies the task:

```python
n = zlib.crc32(f"{run_id}:{item_id}".encode()) % 2**31
amp_seed, base_seed = n, n ^ 0x5BF03635
```

Keying on the work item alone makes items differ but a re-trigger repeat; including the trigger id
makes a re-trigger differ too, while a retry of the same trigger still reproduces. Nothing does this
for you: identical config means identical output, deliberately.

## Credentials

**No credential belongs in this file, ever.** Two paths would leak one: `run()` logs every stage's
argv, and `main()` writes the resolved config into `<output>` — the very tree `publish` uploads.

They come from the environment instead: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (or an
instance role) and `AWS_ENDPOINT_URL` for `s3://`, `MULTISTORAGECLIENT_CONFIGURATION` or
`MSC_CONFIG` for `msc://`, `HF_TOKEN` for Hugging Face, `NGC_API_KEY` for NGC. The amg model
registry is a *path* in the config; the key lives in the file it names.

## staging

```yaml
staging:
  fetch:
    input:                          # a list is an overlay, in order, later wins
      - s3://bucket/shared/assets/  # the bank many runs share
      - /mnt/workitem               # this task's inputs, mounted rather than uploaded
    recipe:     s3://bucket/recipe.yaml
    checkpoint: /mnt/ckpt/model.pt
  publish:
    output: s3://bucket/results/task-42/
```

Every `fetch` key is a **config key**, and that key's value is the destination — so there is no way
to fetch something nothing reads. Sources may be `s3://`, `msc://`, `hf://`, `ngc://`, `usecase://`,
or a local path already on the node. `publish` accepts `s3://` and `msc://` only: Hugging Face and NGC are
where an artifact is published on a human decision, not somewhere a task uploads to by a typo.

A local source must be its **own directory**, never the destination: `fetch` replaces the
destination wholesale, so anything pre-placed there is deleted before the first stage reads it.

`hf://` and `ngc://` forms:

```
hf://datasets/<org>/<repo>[@<revision>][/<subpath>]      hf://models/...
ngc://models/<org>[/<team>]/<name>:<version>[/<subpath>] ngc://resources/...
```

Pin the Hugging Face revision. Unpinned runs, but the log then carries the commit it resolved and a
warning that the source did not name it — a dataset fetched from a branch is not the dataset the
same config fetched last week.

### `usecase://` — a dataset this image knows how to prepare

```yaml
staging:
  fetch:
    input:
      - usecase://magnetic_tile     # runs the shipped script, then lands what it produced
```

A use-case names one of the dataset scripts shipped under `anomalygen/scripts/datasets/`. The script
fetches from wherever that dataset actually lives — a Hugging Face repo, a pinned GitHub archive —
and writes the Step 1 layout, which `prepare` then lands on `input` like any other source. Available
names are `pcb`, `magnetic_tile` and `phone_screen`; a name this image does not ship is refused at
config load, with the ones it does ship listed.

It is a **name, not a path**. The set of runnable scripts is decided by the image, not by the config
— which matters because a config reaches this entrypoint from an orchestrator and is written into
`<output>` verbatim. To run a preparation script of your own, run it yourself and list its output
directory as a local source; that needs nothing from this key.

`usecase://` belongs on **`input`** only. It prepares a whole dataset, and every other fetchable key
names a single file.

It overlays like any other source, so correcting one file of a shipped dataset does not mean forking
the script:

```yaml
    input:
      - usecase://pcb
      - /mnt/my_corrections         # later wins
```

Two things worth knowing before you use one. The script runs inside the `prepare` stage, so its
download happens before `input` is checked — a dataset that takes minutes to fetch takes those
minutes before a typo elsewhere in the config is reported. And a top-level `.cache/` is dropped
rather than landed: every one of these scripts leaves its download cache there (58 MB of the 61 MB
`magnetic_tile` writes), which is the script's working state and not dataset content.

**Parameters, for a dataset that cannot be fully downloaded.** `usecase://<name>?<p>=<source>&<flag>`:
a parameter with a value names a **file to fetch** and hand the script as `--<p> <local path>`; a
bare one is passed as the flag `--<p>`.

```yaml
    input:
      - usecase://phone_screen?zip=s3://bucket/datasets/mobile-screen-coco.zip
```

`phone_screen` is the one that needs this today. Its masks come from Hugging Face, but its clean and
anomaly images come from a Roboflow COCO export that a human downloads through a browser — so the
zip is a **deployment value**: somebody downloads it once, puts it in a bucket or on the node, and
every config afterwards points at the same object. Without it the script refuses and says where the
file comes from; `usecase://phone_screen?masks_only` asks for the mask-only dataset on purpose.

Which parameters a script takes is the script's business, not this key's — there is no list of them
here, because a list here would go stale the first time somebody adds a script without reading this
file. What IS checked before the run starts: the parameter names (each becomes a command-line flag,
so `output_dir` and anything unspellable as one is refused) and that each value is a source `fetch`
can reach.

### `input_override` — generate from these images, not the dataset's

```yaml
staging:
  fetch:
    input:
      - usecase://pcb
    input_override:
      passive_component/clean_image:            # paths INSIDE input, replaced
        - hf://datasets/nvidia/Spark-AnomalyGen-USD@<sha>/input_real_image
      passive_component/cad_mask:
        - s3://bucket/demo/cad_mask/
      defect_spec.jsonl:                        # one path may be one file
        - s3://bucket/demo/defect_spec.jsonl
```

A prepared dataset carries the clean images its author shipped. A run that has to generate from
three specific boards — the ones a demo is about, the ones a customer sent — needs those images and
**not** the shipped ones, so `input_override` REPLACES the path it names rather than adding to it.
Listing the three beside the dataset's own would leave a pool of two hundred and three, all of which
AMP samples.

Each key is a path relative to `input`, and each value is a source list that merges in order, later
winning — the same meaning it has anywhere else in `fetch`. So a texture is just the first segment,
and `<TEXTURE>/mask/<class>`, `defect_spec.jsonl` and `semantic_segmentation_labels.json` are
reachable without a key of their own.

**Individual files, named.** A source naming one file lands in the directory under its own name, so
"generate from THIS board" does not need a prefix that contains only it:

```yaml
    input_override:
      passive_component/clean_image:
        - hf://datasets/nvidia/Spark-AnomalyGen-USD@<sha>/input_real_image/0402_H060.jpg
      passive_component/cad_mask:
        - s3://bucket/demo/0402_H060.png     # the stems must match; see below
```

Named files and prefixes may be mixed in one list. The file keeps the name its source ends in — the
key names the directory, not the file — because a `cad` texture pairs clean image to `cad_mask` by
stem, and a rename under the hood would break that pairing with nothing to point at.

**Keeping the dataset's own name.** A key may name the FILE instead, which is how one image is
swapped without disturbing that pairing: the key supplies the name, the source supplies the bytes.
For a `cad` texture with no new `cad_mask` to go with the new image, this is the form to use:

```yaml
    input_override:
      passive_component/clean_image/C225@1_SolderLight_Missing.jpg:
        - hf://datasets/nvidia/Spark-AnomalyGen-USD@<sha>/input_real_image/0603_H100.jpg
```

Rules, each of which is checked:

- `input_override` needs `input` in the same `fetch` block. It names paths in a tree this run
  fetches; a mounted dataset is the operator's, and replacing a path inside it is not something a
  config should be able to ask for.
- The path must **already be in the fetched input**. `clean_images` for `clean_image` would
  otherwise fetch cleanly, write a directory nobody reads, and generate from the dataset's own
  images as if nothing had been overridden. To ADD a texture, list another source on `input`.
- A file is replaced by exactly one file. A prefix given for `defect_spec.jsonl` would
  replace it with a directory of the same name, and that surfaces stages later as a JSON error.
- The path stays inside `input` — absolute paths and `..` are refused.
- `usecase://` cannot appear here. It prepares a whole dataset; it belongs on `input`.

The replacement happens inside the staging directory, before the merged tree is moved into place, so
`input` is either the previous dataset or the fully overridden one — never, for a moment, the
dataset the config asked not to generate from.

**What the rest of the pipeline still requires.** An override does not relax AMP's preflight. For a
`cad` or `cad_tht` texture every clean image needs a `cad_mask/<stem>.png` of the same size, so
overriding `clean_image` alone fails with `N clean image(s) without matching cad_mask` — override
`cad_mask` too, with files whose stems match. For `text` and `free` textures the clean images stand
alone.

## Every key

<!-- BEGIN generated key table -->

| key                          | type                                             | default                                               | meaning                                                                                                                                                                                                                                                                                          |
| ---------------------------- | ------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `input`                      | string                                           | **required**                                          | Dataset root in the Step 1 layout: <TEXTURE>/{clean_image,cad_mask,mask/<defect>}/ plus defect_spec.jsonl at the root. Under `prepare` this is where staging.fetch.input lands, and anything already there is deleted by the swap.                                                               |
| `output`                     | string                                           | **required**                                          | Every stage writes under here. Created if absent.                                                                                                                                                                                                                                                |
| `recipe`                     | string                                           | **required**                                          | The experiment recipe the fine-tuned checkpoint was trained with. Its model_size selects which base checkpoint generate loads.                                                                                                                                                                   |
| `checkpoint`                 | string \| null                                   | `null`                                                | The FINE-TUNED weights. Required by generate and refine; ignored by a mask-only run.                                                                                                                                                                                                             |
| `base_checkpoint`            | string \| null                                   | `null`                                                | The frozen base DCP. null lets generate pick it from the recipe's model_size, which is what almost every run wants.                                                                                                                                                                              |
| `num_sdg`                    | integer \| null                                  | `null`                                                | How many images to generate. Required when a mask stage runs. Ignored when per_defect_counts is given, whose sum wins.                                                                                                                                                                           |
| `per_defect_counts`          | object \| null                                   | `null`                                                | {defect_type: count}. Selects which types to place and how many of each; omit to use every type in defect_spec.jsonl. Requires mode: inference.                                                                                                                                                  |
| `mode`                       | string — one of `"inference"`, `"validation"`    | `"inference"`                                         | inference allocates uniformly over the selected types; validation allocates proportionally over every type and refuses per_defect_counts.                                                                                                                                                        |
| `amp_per_clean_pairing`      | string — one of `"on"`, `"off"`, `true`, `false` | `"off"`                                               | amp only. on gives every clean image exactly one output and accepts exactly one defect type -- what a fan-out over work items needs. off round-robins, maximising (submask, clean) variety without covering every clean image. YAML parses bare on/off as booleans; both spellings are accepted. |
| `guidance`                   | number \| null                                   | `null`                                                | Classifier-free guidance. null uses the shared texture default.                                                                                                                                                                                                                                  |
| `num_steps`                  | integer \| null                                  | `null`                                                | Diffusion steps. null uses the shared texture default.                                                                                                                                                                                                                                           |
| `crop_ratio`                 | number \| null                                   | `null`                                                | How much context around the ROI the crop keeps. null uses the shared texture default.                                                                                                                                                                                                            |
| `amp_seed`                   | integer                                          | `42`                                                  | amp only: which base mask is drawn and where it is placed. No effect under amg, where the model decides placement.                                                                                                                                                                               |
| `base_seed`                  | integer                                          | `1`                                                   | The diffusion noise. A different roll of the dice, not a different placement -- changing it alone re-renders the identical mask.                                                                                                                                                                 |
| `num_search_run`             | integer                                          | `3`                                                   | refine only: how many candidates each search round generates.                                                                                                                                                                                                                                    |
| `num_gpus`                   | integer                                          | `1`                                                   | Ranks for generate and refine. Checked against the visible GPUs before any stage runs.                                                                                                                                                                                                           |
| `guardrail`                  | boolean                                          | `true`                                                | Content-safety filtering in generate. Off skips the blocklist and face-blur weights entirely.                                                                                                                                                                                                    |
| `output_naming`              | string — one of `"source"`, `"indexed"`          | `"source"`                                            | source keeps the source crop's own filename on the result, which is what a compositing step needs; it requires one output per clean image. indexed names by <defect_type>_<NNNNN>.                                                                                                               |
| `amg_gen_model`              | string \| null                                   | `null`                                                | amg only: registry name of the model that proposes and refines masks. Required under amg unless amg_serve supplies it.                                                                                                                                                                           |
| `amg_judge_model`            | string \| null                                   | `null`                                                | amg only: registry name of the model that scores candidates. Required under amg unless amg_serve supplies it. The judge matters more than the generator.                                                                                                                                         |
| `amg_model_registry`         | string \| null                                   | `null`                                                | amg only: path to the model registry file ({model, base_url, key} entries). Required under amg unless amg_serve is set. NEVER put the key itself in this config.                                                                                                                                 |
| `amg_cache_dir`              | string \| null                                   | `null`                                                | amg only: the reusable distilled defect library. null puts it under <output>/amg_cache; point several runs of one dataset at a shared directory to reuse it.                                                                                                                                     |
| `amg_serve`                  | string \| null                                   | `null`                                                | amg only: a serving recipe run_pipeline starts before the stage and stops after. Set it instead of amg_model_registry when nothing is serving yet; it also defaults the two model names.                                                                                                         |
| `amg_beam_multi`             | integer                                          | `0`                                                   | amg only: in-loop beam width. 0 is off. A small local model needs it; a strong cloud model does not.                                                                                                                                                                                             |
| `amg_beam_pairwise`          | boolean                                          | `false`                                               | amg only: pairwise comparison in the beam.                                                                                                                                                                                                                                                       |
| `amg_exemplar_captions`      | boolean                                          | `false`                                               | amg only: feed exemplar captions to the generator.                                                                                                                                                                                                                                               |
| `amg_exemplar_visual_prompt` | boolean                                          | `false`                                               | amg only: feed exemplar images to the generator.                                                                                                                                                                                                                                                 |
| `amg_coverage_note`          | boolean                                          | `false`                                               | amg only: feed a coverage note to the generator.                                                                                                                                                                                                                                                 |
| `staging`                    | object \| null                                   | `null`                                                | Where prepare fetches from and publish uploads to. NEVER put a credential here: the resolved config is written into <output>, which publish then uploads.                                                                                                                                        |
| `stages`                     | array                                            | `["amp", "generate", "evaluate", "refine", "export"]` | Which stages run, in any order -- they are reordered to the canonical one and deduplicated. A skipped stage's artifacts must already be under `output`, which is how a run resumes.                                                                                                              |

<!-- END generated key table -->

## Worked shapes

**A batch.** Many crops in, many images out, one run.

```yaml
num_sdg: 20
per_defect_counts: {SolderPin+bridge: 10, SolderPin+trapped: 10}
amp_per_clean_pairing: off
output_naming: indexed
stages: [prepare, amp, generate, export, publish]
```

**A fan-out task.** One crop in, one image out, one run per crop — `amp_seed` and `base_seed`
derived per task as above.

```yaml
num_sdg: 1
per_defect_counts: {SolderPin+bridge: 1}
amp_per_clean_pairing: on
output_naming: source          # the result keeps the crop's own filename
stages: [prepare, amp, generate, export, publish]
```

**The agentic backend, against a model that is already serving.** A cloud API here; the same shape
points at a shared vLLM the orchestrator runs as a service. The pod starts nothing and needs
`num_gpus` GPUs only.

```yaml
num_sdg: 6
amg_serve: null
amg_gen_model: azure/openai/gpt-6-astra
amg_judge_model: azure/openai/gpt-6-astra
amg_model_registry: /work/endpoints.yaml   # fetched below; holds the key, so it never lives in this file
staging:
  fetch:
    amg_model_registry: s3://YOUR-BUCKET/secrets/endpoints.yaml
stages: [prepare, amg, generate, export, publish]
```

**The agentic backend, serving its own model.** Nothing needs to be serving beforehand; the pod
needs `max(tp, num_gpus)` GPUs -- 2 for this recipe -- and the model's weights on the checkpoints
volume.

```yaml
num_sdg: 7
per_defect_counts: {passive_component+missing: 7}
amg_serve: gemma-4-31b-it      # started before the stage, stopped after
amg_beam_multi: 4              # a local model needs the in-loop beam
amg_beam_pairwise: true
stages: [prepare, amg, generate, export, publish]
```
