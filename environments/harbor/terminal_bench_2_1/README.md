# Terminal Bench 2.1 on the Harbor task shape

The 89 Terminal Bench 2.1 tasks run as published Harbor task folders through the generic `harbor` environment
server, with no Gym-side patching; the commands and the known upstream noise are documented in
[Evaluate Terminal-Bench 2.1 on the Harbor Path](https://docs.nvidia.com/nemo/gym/main/tutorials/evaluation-tutorials/terminal-bench-2-1-harbor).
The legacy benchmark folder, `benchmarks/terminal_bench_2_1/`, is untouched.

`dataset.toml` here pins the dataset's source, Harbor's package store copy by content hash (the store tags it
`latest` only). `gym dataset pull terminal_bench_2_1` fetches the 89 task folders into this folder (gitignored
beside this README), and `gym dataset validate terminal_bench_2_1 --sandbox opensandbox` and
`gym eval run terminal_bench_2_1 ...` resolve the entry name to it.
