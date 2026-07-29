from nemo_gym.decorators import experimental
from nemo_gym.orchestration.api import SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.slurm import SlurmExecutor

_EXECUTORS = {
    SlurmComputeConfig: SlurmExecutor,
}


@experimental
def submit(config: SubmitConfig, *, dry_run: bool = False) -> None:  # pragma: no cover
    compute = next(iter(config.compute.values()))
    _EXECUTORS[type(compute)]().run(config, dry_run=dry_run)
