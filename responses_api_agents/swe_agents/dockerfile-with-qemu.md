```bash
docker buildx build \
  --progress=plain \
  --platform linux/arm64 \
  -f responses_api_agents/swe_agents/Dockerfile.with_qemu \
  -t gitlab-master.nvidia.com:5005/nexus-team/nexusnest/container_with_qemu:test001 .
```
