from nemo_gym.trajectory_collection import TrajectoryCollectionConfig


# TODO: Eventually we want to add more tests to ensure that the trajectory collection flow does not break
class TestTrajectoryCollection:
    def test_sanity(self) -> None:
        TrajectoryCollectionConfig(
            agent_name="",
            input_jsonl_fpath="",
            output_jsonl_fpath="",
        )
