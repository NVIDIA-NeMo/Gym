from responses_api_agents.mini_swe_agent.sandbox_environment import MiniSWESandboxEnvironment, Submitted


def test_check_finished_raises_submitted_for_submit_sentinel() -> None:
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    try:
        env._check_finished(
            {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
                "returncode": 0,
                "exception_info": "",
            }
        )
    except Submitted as error:
        assert error.messages == (
            {
                "role": "exit",
                "content": "patch contents\n",
                "extra": {"exit_status": "Submitted", "submission": "patch contents\n"},
            },
        )
    else:
        raise AssertionError("Expected Submitted")


def test_check_finished_ignores_nonzero_submit_sentinel() -> None:
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    env._check_finished(
        {
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
            "returncode": 1,
            "exception_info": "",
        }
    )
