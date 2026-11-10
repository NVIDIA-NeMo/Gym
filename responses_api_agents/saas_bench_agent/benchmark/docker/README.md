# Docker Images

SaaS-Bench evaluates agents against 14 self-hosted SaaS applications running
inside Docker. We distribute these as pre-built `.tar` archives so users do
not need to build images themselves.

## 1. Download

Download all image archives from Huggingface: https://huggingface.co/datasets/Marti844/SaaS-Bench-docker

Place the `.tar` files (or `.tar.gz`) under this directory:

```
docker/images/
├── mw-code-server.tar
├── mw-openproject.tar
├── mw-metabase.tar
├── mw-baserow.tar
├── mw-twenty.tar
├── mw-bigcapital.tar
├── mw-hrms.tar
├── mw-pretix.tar
├── mw-openemr.tar
├── mw-opnform.tar
├── mw-onlyoffice.tar
├── mw-mattermost.tar
├── mw-owncloud.tar
└── mw-roundcubemail.tar
...
```

## 2. Load

From the repository root:

```bash
bash scripts/load_images.sh
```

This will `docker load` every archive in `docker/images/`. Verify with:

```bash
docker images | grep '^mw-'
```

You should see 14 `mw-*:latest` images.

## 3. Compose-based applications

Four applications run as multi-container `docker compose` stacks:

- `pretix`     → `pretix.yml.tpl`
- `onlyoffice` → `onlyoffice.yml.tpl`
- `mattermost` → `mattermost.yml.tpl`
- `owncloud`   → `owncloud.yml.tpl`

These templates are instantiated per slot at runtime by `saas_bench/slot.py`.
Auxiliary images (postgres, redis, etc.) referenced inside the templates are
pulled from Docker Hub on first run; make sure the host has internet access
the first time you launch them, or pre-pull them yourself.

## 4. Disk usage

The full image set is roughly **~60 GB** uncompressed. Allow ~100 GB free on
the partition holding `/var/lib/docker` before loading.

