Note: Currently the branch is checked out from the `text_based_game` branch (which is for the sudoku game) since I need the `simple_game_agent.py` (added in the `text_based_game` branch) to test. Once that branch is merged these extra diffs will vanish.

## Details 
This is a resource server for minesweeper text based game. The correctness is checked programtically. The reward is returned at each step (each call of the `make_move` function. The final reward is the sum)

**Dataset**: The dataset is proceduraly generated. You specify a `rows`, `cols`, `num_mines` and `max_turns` and the game generates the initial board.
 
For it to be compatible with current trajectory generation framework. We have a `simple_minesweeper.jsonl` that can be used to for trajectory generation. This would be changed later

## How to run

1. Download dataset

```
ng_download_dataset_from_gitlab \
    +dataset_name=simple_minesweeper \
    +version=0.0.1 \
    +artifact_fpath=simple_minesweeper.jsonl \
    +output_fpath=data/simple_wordle.jsonl
```

2. Start servers

```
ng_run "+config_paths=[responses_api_agents/simple_game_agent/configs/simple_game_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml,resources_servers/simple_minesweeper/configs/simple_minesweeper.yaml]" +simple_game_agent.responses_api_agents.simple_game_agent.resources_server.name=simple_minesweeper
```

3. Run trajectory collection
```
```

### Collected trajectory examples
```
```