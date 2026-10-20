# From a request to a run_pipeline config

How to turn a request such as *"Run Day 1 for component 0603_H100 to generate missing defects"*
or *"Run Day 0 for component 115-2819-000 to generate bridge defects"* into the YAML `anomalygen.scripts.run_pipeline` takes. Written for whoever -- or whatever -- writes
that YAML on the workflow side: it states the decisions, the rules that constrain them, and the
reference files a generated config must match.

Companions: `config.md` (beside this file) is the key-by-key specification (read it for what a key means);
`ag_config/pipeline/templates/*.yaml` are the reference configs this file is written against; and

```shell
python -m anomalygen.scripts.run_pipeline --config <file> --dry_run
```

is the only validator. Whatever it says is what the run will say.

## Which use case

Three fine-tuned models ship, one per use case, and a request is always about one of them. Each has
its own template family, its own dataset script and its own trained defect types:

| use case        | the request says                                         | texture                   | trained defect types                                          | dataset                                                               | templates                 |
| --------------- | -------------------------------------------------------- | ------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------- |
| `pcb`           | PCB, a component name (`0603_H100`, `115-2819-000`, ...) | `IC`, `passive_component` | `bridge`; `excess_solder`, `missing`                          | `usecase://pcb`, plus the request's own image(s) via `input_override` | `pcb_day{0,1}_*.yaml`     |
| `magnetic_tile` | metal surface, magnetic tile                             | `metal_surface`           | `MT_Blowhole`, `MT_Break`, `MT_Crack`, `MT_Fray`, `MT_Uneven` | `usecase://magnetic_tile`, its own clean images                       | `magnetic_tile_day1.yaml` |
| `phone_screen`  | glass, phone screen                                      | `Phone`                   | `oil`, `scratch`, `stain`                                     | `usecase://phone_screen?zip=...`, its own clean images                | `phone_screen_day1.yaml`  |

pcb is the one where the request brings the image and names a component; Decisions 0 to 3 below are
about it. magnetic_tile and phone_screen generate on the clean images their datasets ship, so for them there is no
image to place and no component: the request names the use case and, optionally, which of its
defect types. Their files are `_day1` because that is the flow they run in -- the model generates
on clean images that already exist -- and they have no Day 0, since nothing simulates a tile or a
screen; the day is not a decision for them.

**Narrowing a magnetic_tile / phone_screen request.** The template lists every trained type, five images each, and
`num_sdg` is their sum. *"Run the day 1 workflow to generate images of metal surface defects"* keeps
the template as is. *"... generate blowhole and crack defects on metal surface"* keeps only
`metal_surface+MT_Blowhole` and `metal_surface+MT_Crack`, sets `num_sdg: 10`, and changes nothing
else. A type outside the table -- *rust on metal surface* -- is refused the same way a pcb request for
an untrained pair is. Spellings are exact: `MT_Blowhole` with the prefix, `oil` lower-case.

phone_screen's dataset script cannot download its images: they are a Roboflow export a person downloads once,
by hand, following the ten steps under *Phone screen* in `docs/tutorial/texture/2-dataset-preparation.md`
(a free Roboflow account, fork the project, export all 320 items as COCO, download the zip). That zip
then goes into a bucket the pod can read, and its URL is the `?zip=` value in the template -- a
`DEPLOYMENT` value like the mounts, set once. **If the deployment has not supplied that URL, do not
write a config**: there is no default and nothing to invent; hand the requester those steps instead.
Without it `prepare` refuses the run and points at the same page.
magnetic_tile needs a network reach to GitHub, nothing else.

`magnetic_tile` is also the one template with `guardrail: false`. With the guardrail on, the text
check reads the word "crack" in every `MT_Crack` caption as an illegal act and blocks all five images
-- zero delivered, and the run still passes every gate -- so a request for crack defects would
publish nothing and report success. The dataset is industrial surface imagery in which nothing the
guardrail protects can appear; the template turns it off and says so. Do not copy that line into the
other templates.

## Decision 0: Day 0 or Day 1 (pcb)

The request says which. They differ in where the clean images come from and, because of that, in
the model that places the masks; they share everything else.

|                 | Day 1                                                        | Day 0                                                                                                                                                |
| --------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| clean images    | **one real photo** of the component, attached to the request | the **simulated crops** of the component that the workflow's upstream tasks produced (CAD render, then sim2real augmentation); several per component |
| how they arrive | one file URL (`hf://` pinned, or `s3://`)                    | one `s3://` prefix the workflow wrote, **flat**, every file uniquely named                                                                           |
| mask model      | Astra through a registry file (`amg_serve: null`)            | Gemma started by the run (`amg_serve: gemma-4-31b-it`)                                                                                               |
| pod             | `num_gpus` GPUs                                              | `max(2, num_gpus)` GPUs and the Gemma weights on the checkpoints volume                                                                              |
| templates       | `pcb_day1_*.yaml`                                            | `pcb_day0_*.yaml`                                                                                                                                    |

The model choice is fixed per template family, not a decision: the cloud default is set by policy
for the release, the Gemma family by measurement on simulated crops. A request does not pick it.

## What a request contains, and what it does not

A request names **one component**, points at its **clean images** (Day 1: one real image; Day 0: the
prefix of its simulated crops), and names **one defect type**. Everything else about the run is
fixed by the use case (pcb) and the day, and is not the request's to change. So a generated
config differs from the template for its day and defect type in exactly four values, each marked
`CHANGES PER REQUEST` in the template:

| line                                         | what goes there                                                                                                               |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `per_defect_counts: {<texture>+<defect>: N}` | the one type this request asks for, and how many images                                                                       |
| the `staging.fetch.input_override` key       | `<texture>/clean_image` -- Decision 1 is the first segment                                                                    |
| that key's one source                        | Day 1: the attached image, a pinned `hf://` or `s3://` file URL. Day 0: the `s3://` prefix of the component's flattened crops |
| `staging.publish.output`                     | where this request's results go                                                                                               |

Every other line is copied from the template as is -- with one more distinction the templates mark:

| line kind             | who sets it                          | when it changes                              | examples                                                                                                        |
| --------------------- | ------------------------------------ | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| unmarked              | this repo: facts about the pcb model | when the model changes                       | `recipe`, the prepare script, the model names, `stages`                                                         |
| `DEPLOYMENT`          | the workflow: its pods and buckets   | once per deployment, shared by every request | `input`/`output` volume, `checkpoint` and `amg_model_registry` mount paths, `amg_cache_dir`, the publish bucket |
| `CHANGES PER REQUEST` | the generator, from the request      | every request                                | `per_defect_counts`, the override, the publish tail (pcb only)                                                  |

The `DEPLOYMENT` values in the templates are the ones our runs used; a deployment replaces them once
and then treats them as fixed. A generated config may differ from its template only on marked lines.

## Decision 1: component → texture

The fine-tuned pcb model knows two textures. A component is one or the other by what it is:

| texture             | components                                          | how to recognise them                                                                                       |
| ------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `passive_component` | chip passives: resistors, capacitors, inductors     | package-size names: `0201_*`, `0402_*`, `0603_*`, `0805*`, `1206*`, `1210*` (e.g. `0603_H100`, `0402_H060`) |
| `IC`                | integrated circuits, connectors, other leaded parts | part-number names: `NNN-NNNN-NNN` (e.g. `115-2819-000`, `130-1534-000`)                                     |

Component names arrive in several spellings: `_115-2819-000`, `115-2819-000_1`, `115_2819_000`.
Strip a leading underscore and a trailing `_<n>` instance suffix; treat `-` and `_` as the same
character. The normalised name (`115-2819-000`) is what goes into the publish tail; the image file
keeps whatever name it arrived with.

If a component matches neither pattern, ask; do not guess a texture.

## Decision 2: does the texture support the defect?

The model was trained on exactly three `<texture>+<defect>` pairs
(`ag_config/exp_texture_ft_pcb.yaml`, `anomaly_types`):

| request says  | texture it needs    | valid for          |
| ------------- | ------------------- | ------------------ |
| bridge        | `IC`                | ICs only           |
| excess solder | `passive_component` | chip passives only |
| missing       | `passive_component` | chip passives only |

A request that crosses this table -- *bridge on 0603_H100*, *missing on 115-2819-000* -- has no
valid config. **Refuse it and say why**; do not substitute a texture the component is not, and do
not substitute a defect the request did not ask for. The run would refuse it too
(`per_defect_counts` names a type not in `defect_spec.jsonl`), but only after fetching everything.

Spelling: the key is `<texture>+<defect>` with the names exactly as above -- `excess_solder` with an
underscore, `missing` alone (not `missing_component`).

## Decision 3: how many images

`num_sdg` and the count in `per_defect_counts` are the same number and must agree. The templates
use 5. A request that names a count uses it; one that does not, uses 5. `amg` draws clean images
round-robin -- Day 1 has one, so it is reused; Day 0 has several, so the first five distinct ones
are used once each -- and `output_naming` is `indexed` either way. `amp_per_clean_pairing` stays
`off`. Both are fixed in the template, not decisions.

## Where the image goes: `input_override`

`prepare` builds the dataset from `usecase://pcb` -- pcb's shipped script
(`anomalygen/scripts/datasets/prepare_pcb_defect.py`), which pulls `nvidia/Cosmos-AnomalyGen-PCB-Dataset`
into the layout every stage reads (`<texture>/{mask,anomaly_image,clean_image}/`, `defect_spec.jsonl`).
That dataset carries its own clean images, and `amg` generates on every clean image under a
texture, so the request's image must **replace** them, not join them. `input_override` does that:
each key is a path inside `input`, and the path is replaced by the sources listed for it.

```yaml
staging:
  fetch:
    input:
      - usecase://pcb
    input_override:
      passive_component/clean_image:                                        # Decision 1 is the first segment
        - hf://datasets/nvidia/Spark-AnomalyGen-USD@0bbac7b7322afcfa0d58c837c17fddaee84d9b76/input_real_image/0603_H100.jpg
```

After `prepare`, `<input>/passive_component/clean_image/` holds exactly that one file, under its own
name; every other texture and every mask bank is untouched. A request writes exactly one key with
exactly one source.

For Day 0 the one source is a **prefix**, and the workflow owns its contents:

```yaml
    input_override:
      IC/clean_image:
        - s3://<bucket>/.../runs/<run>/augment/115-2819-000/IC/
```

Everything under that prefix lands in `clean_image/` **with its sub-path kept**, and `amg` reads the
directory's top level only and refuses two files with the same stem. So the upstream augment task
must write the crops **flat and uniquely named** (e.g. `x1_y1__0001.jpg`, cell and index joined);
the config neither knows nor checks how many there are. A nested layout (`x1_y1/0001.jpg`) or
repeated stems is an upstream defect, not something to fix in the config.

Two forms exist and the choice is not free:

| key names                                             | effect                                                                               | when                                                                                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| the **directory** `<texture>/clean_image`             | the directory is emptied and refilled with the listed files, each under its own name | **Day 1 under `amg`** -- the only image left is the attached one, and its name is what the export manifest reports                      |
| a **file** `<texture>/clean_image/<dataset stem>.jpg` | that one dataset file is replaced, keeping the dataset's stem                        | only for `amp` on a `cad`-routed texture, where the stem has to keep matching its `cad_mask`; the other shipped images stay in the pool |

Pin the revision in the URL (`@<sha>`, as above). `@main` runs, but the log will say the source did
not name the commit it resolved. The key must exist in the prepared dataset; a misspelling
(`clean_images`) is refused at `prepare` rather than silently creating an unread directory.

The three Spark real images and what they are:

| file                                | component      | texture                                                         |
| ----------------------------------- | -------------- | --------------------------------------------------------------- |
| `input_real_image/115_2819_000.jpg` | `115-2819-000` | `IC`                                                            |
| `input_real_image/0603_H100.jpg`    | `0603_H100`    | `passive_component`                                             |
| `input_real_image/0402_H060.jpg`    | `0402_H060`    | `passive_component` (217x130, half the training crop; untested) |

## What is fixed, and why

| block                              | value                                                                                                                           | why it is not a decision                                                                                                                           |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| dataset                            | `usecase://pcb` as the `input` source                                                                                           | pcb's own prepare script; it pulls the bank the model was trained against                                                                          |
| recipe                             | `ag_config/exp_texture_ft_pcb.yaml`, from the code tree                                                                         | the pcb model trained under it; class ids must line up                                                                                             |
| checkpoint                         | `/mnt/ckpt/pcb/iter_000015000.pt`, mounted (an `hf://` source once published)                                                   | the pcb fine-tuned weights                                                                                                                         |
| model, pcb Day 1 and phone_screen  | Astra through a registry file (`amg_serve: null`, both model names `azure/openai/gpt-6-astra`)                                  | the cloud default; the pod needs `num_gpus` GPUs only                                                                                              |
| model, pcb Day 0 and magnetic_tile | Gemma started by the run (`amg_serve: gemma-4-31b-it`, model names null, the four local-model channels on, `amg_beam_multi: 4`) | best measured on simulated crops; the pod needs `max(2, num_gpus)` GPUs and `google/gemma-4-31b-it` on the checkpoints volume; no registry, no key |
| `amg_model_registry`               | Day 1: `/mnt/secrets/endpoints.yaml`, mounted. Day 0: `null`                                                                    | the file holds the API key; the config is uploaded with the results, so the key never goes in it                                                   |
| `amg_cache_dir`                    | `/work/amg_cache/pcb`                                                                                                           | a writable volume path; the distilled defect library is reused across runs of the same use case                                                    |
| `stages`                           | `[prepare, amg, generate, export, publish]`                                                                                     | Day 1 is generation; `evaluate` and `refine` need the KPI and are not part of it                                                                   |
| `num_gpus`                         | 1                                                                                                                               | one image, one GPU                                                                                                                                 |
| `guardrail`                        | `true`; `false` for `magnetic_tile` only                                                                                        | content safety stays on                                                                                                                            |

The two model blocks are the two template families; a request never mixes them.

## The templates

| request                                                         | template                                                                                            |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Day 1, bridge defects for `<component>` (an IC)                 | `ag_config/pipeline/templates/pcb_day1_bridge.yaml`                                                 |
| Day 1, excess-solder defects for `<component>` (a chip passive) | `ag_config/pipeline/templates/pcb_day1_excess_solder.yaml`                                          |
| Day 1, missing defects for `<component>` (a chip passive)       | `ag_config/pipeline/templates/pcb_day1_missing.yaml`                                                |
| Day 1, bridge defects for `115-2819-000` with its real image    | `ag_config/pipeline/templates/pcb_day1_115-2819-000_bridge.yaml`                                    |
| magnetic_tile, all or some metal-surface defect types           | `ag_config/pipeline/templates/magnetic_tile_day1.yaml`                                              |
| phone_screen, all or some phone-screen defect types             | `ag_config/pipeline/templates/phone_screen_day1.yaml`                                               |
| Day 1, missing defects for `0603_H100`                          | `ag_config/pipeline/templates/pcb_day1_0603_H100_missing.yaml`                                      |
| Day 0, the same three shapes                                    | `ag_config/pipeline/templates/pcb_day0_{bridge,excess_solder,missing}.yaml`                         |
| Day 0, `115-2819-000` bridge; `0603_H100` missing               | `ag_config/pipeline/templates/pcb_day0_115-2819-000_bridge.yaml`, `pcb_day0_0603_H100_missing.yaml` |

Of the pcb files, three per day carry `<component>` (and, for Day 0, `<run>`) placeholders; the two
per day named after a component are those filled in, and run `--dry_run` as they are. The magnetic_tile and
phone_screen files carry no placeholder at all.

**One publish rule.** Results publish under the deployment's prefix root followed by
`<day>/<usecase>/`; for pcb, and only pcb, `<component>/<defect>/` follows, because a pcb request
names both and nothing else distinguishes two of them. The other use cases stop at the use-case name
-- a request that narrowed the types still publishes there, and the file names carry the type
(`metal_surface+MT_Break_00000.png`). So:

```
.../defect_image_generation/day1/pcb/0603_H100/missing/
.../defect_image_generation/day0/pcb/115-2819-000/bridge/
.../defect_image_generation/day1/magnetic_tile/
.../defect_image_generation/day1/phone_screen/
```

A generated config that is not byte-equal to its template outside the marked values is wrong until
shown otherwise.

## Checking a generated config

1. `--dry_run` it. It validates every key and every cross-key rule, checks the registry names the
   models, and prints `gpus_required` and `service:`. What to expect follows the model block, not
   the day: a registry template (pcb Day 1, phone_screen) prints `gpus_required: 1` and
   `service: none`; a Gemma-serving template (pcb Day 0, magnetic_tile) prints `gpus_required: 2`
   and `service: gemma-4-31b-it.sh (tp 2, port 8902)`.
2. Diff it against the template for its defect type. Only `DEPLOYMENT` and `CHANGES PER REQUEST`
   lines may differ, and the `DEPLOYMENT` lines must equal the deployment's own values, not vary
   between requests.
3. Read `per_defect_counts` back against Decision 2.

What `--dry_run` cannot check: that the override's source resolves (it is fetched by `prepare`,
which has not run yet). A URL that does not resolve fails `prepare`. A key whose texture does not
match the `per_defect_counts` key leaves the generating texture with the dataset's own clean
images and the request's image under a texture nobody generates on -- `amg` then reports the
dataset's images, not the attached one. Decision 1 and the first segment of the key must be the
same value.
