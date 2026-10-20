# Running the container

Only when the user asks to run it — building and pushing do not require this. Assumes the `IMAGE`
variable from the shared variables in `SKILL.md`.

Bind-mount the repo over the workdir; `--user` keeps outputs host-owned; `--shm-size` feeds the
dataloader workers. `-e USER` is what lets `getpass.getuser()` resolve the numeric UID: its lookup
tries the `LOGNAME`, `USER`, `LNAME` and `USERNAME` variables before falling back to the passwd
database, so one env var is enough and the container never needs the host user list.

```shell
docker run --rm -it --gpus all --shm-size=16g \
    --user $(id -u):$(id -g) -e USER=$(id -un) -e HOME=/tmp \
    -v "$PWD:/workspace/cdf-anomalygen" \
    -w /workspace/cdf-anomalygen "$IMAGE" bash
```

**Downloading from the Hugging Face Hub** — only then, add the credential explicitly. It is readable
by every process in the container, so scope it to read-only and leave it off any run that doesn't
need it:

```shell
#   ... -e HF_TOKEN "$IMAGE" bash        # grants the container your Hub access
```

If a library looks the UID up in the passwd database directly instead of going through
`getpass.getuser()`, it raises `KeyError: getpwuid(): uid not found` — the image bakes user
`anomalygen` at UID 10000, which won't match your host UID. Build the image with your own IDs rather
than exposing the host's account data to the container:

```shell
docker build ... --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g) ...
```

Air-gapped images need **no** checkpoint mount (baked in). Step 2 saved the bundle to
`./results/…tar.gz` — copy it to the offline host, `docker load < <bundle>.tar.gz`, then run as above.
