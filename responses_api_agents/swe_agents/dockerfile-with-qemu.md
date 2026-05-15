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

Allow linux/arm64 run on x86
```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

Run
```bash
docker run \
    --platform linux/arm64 \
    gitlab-master.nvidia.com:5005/nexus-team/nexusnest/container_with_qemu \
    uname -m
```
