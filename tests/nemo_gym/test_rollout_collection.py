from nemo_gym.rollout_collection import RolloutCollectionConfig


# TODO: Eventually we want to add more tests to ensure that the rollout collection flow does not break
class TestRolloutCollection:
    def test_sanity(self) -> None:
        RolloutCollectionConfig(
            agent_name="",
            input_jsonl_fpath="",
            output_jsonl_fpath="",
        )
