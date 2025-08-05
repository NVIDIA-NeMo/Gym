from nemo_gym.trajectory_collection import TrajectoryCollectionConfig


class TestTrajectoryCollection:
    def test_sanity(self) -> None:
        TrajectoryCollectionConfig(
            agent_name="",
            input_jsonl_fpath="",
            output_jsonl_fpath="",
        )
