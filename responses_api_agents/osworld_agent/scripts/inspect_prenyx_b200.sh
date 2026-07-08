#!/usr/bin/env bash
# Read-only Prenyx discovery. Run on login-prenyx before submitting anything.

set -euo pipefail

echo "cluster: $(hostname -f)"
echo "slurm:   $(sinfo --version)"
echo
echo "B200-capable partitions/nodes:"
sinfo -N -h -o '%P|%N|%t|%G|%C' | awk 'BEGIN{IGNORECASE=1} /b200/' | sort -u
echo
echo "Current user's associations:"
sacctmgr -n -P show assoc where user="${USER}" format=Cluster,Account,User,Partition,QOS 2>/dev/null || true
echo
echo "Current user's jobs:"
squeue -u "${USER}" -o '%.18i %.12P %.28j %.8T %.10M %.20R'
