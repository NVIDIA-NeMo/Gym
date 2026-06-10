apt-get update
apt-get install -y qemu-system
apt install qemu-user-static binfmt-support

apptainer run docker://multiarch/qemu-user-static --reset -p yes
apptainer pull --arch arm64 docker://multiarch/qemu-user-static
apptainer run qemu-user-static_latest.sif --reset -p yes
apptainer run results/swebench_verified_containers/swebench_sweb.eval.x86_64.astropy_1776_astropy-12907.sif uname -m
