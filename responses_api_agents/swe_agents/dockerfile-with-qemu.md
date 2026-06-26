Allow linux/arm64 build on x86 machine
```bash
docker buildx create --use --name multiarch
docker buildx inspect --bootstrap
```

Build
```bash
docker buildx build \
  --progress=plain \
  --platform linux/arm64 \
  -f responses_api_agents/swe_agents/Dockerfile.with_qemu \
  --load \
  -t gitlab-master.nvidia.com:5005/nexus-team/nexusnest/container_with_qemu .
```

Push
```bash
docker push gitlab-master.nvidia.com:5005/nexus-team/nexusnest/container_with_qemu:latest
```

Pull via enroot
```bash
CONTAINER_IMAGE_PATH="results/container_with_qemu.sqsh"
enroot import -o "$CONTAINER_IMAGE_PATH" 'docker://gitlab-master.nvidia.com#nexus-team/nexusnest/container_with_qemu:latest'
```

Pull one apptainer
```bash
apptainer pull docker://swebench/sweb.eval.x86_64.astropy_1776_astropy-12907
```

Allow linux/arm64 run on x86
```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

Run container interactively
```bash
docker run \
    --platform linux/arm64 \
    -v $(pwd):/workdir \
    -it \
    gitlab-master.nvidia.com:5005/nexus-team/nexusnest/container_with_qemu \
    /bin/bash
```

```bash
update-binfmts --enable qemu-aarch64
update-binfmts --enable qemu-x86_64

apptainer run /workdir/sweb.eval.x86_64.astropy_1776_astropy-12907_latest.sif uname -m
```
