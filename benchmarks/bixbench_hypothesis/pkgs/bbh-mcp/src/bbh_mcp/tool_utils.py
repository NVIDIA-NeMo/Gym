RUBRIC_SCORE_PROMPT = """
Your task is to fairly and accurately score a solution to a bioinformatics task.

The task was to substantiate or reject a hypothesis given a dataset. It was executed
by writing a Jupyter notebook to analyze the provided data.

Here is the hypothesis: {hypothesis!r}.

The hypothesis is known to be {accepted}. You are to score both the final outcome and the procedure used to arrive at it. Use the following rubric:
<rubric>
{rubric}
</rubric>

Now, I will provide you with the solution. First, the notebook:
<notebook>
{notebook}
</notebook>

The final conclusion derived from the notebook:
<proposed-solution>
{proposed_solution}
</proposed-solution>

Be scientifically rigorous. Reason through each criterion in the rubric and provide brief justification for your score. Do not assign partial score for any rubric item (i.e. 0.5 points).
At the very end, provide an integer score based on the rubric, enclosed in <score>...</score> tags.
""".strip()


def list_dir_tool(directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
    from pathlib import Path

    try:
        max_files = int(max_files)
    except (TypeError, ValueError, OverflowError):
        max_files = 20

    show_hidden = bool(show_hidden)

    try:
        directory_path = Path(directory).expanduser().resolve()
        if not directory_path.is_dir():
            return f"Path is not a directory: {directory}"
        if ".harbor" in directory_path.parts:
            return "Directory is empty."

        def collect_files(path: Path, prefix: str = "") -> list[str]:
            try:
                entries = sorted(path.iterdir(), key=lambda entry: (entry.is_file(), entry.name))
            except PermissionError:
                return [f"# {prefix}{path.name}/ (permission denied)"]

            files: list[str] = []
            for entry in entries:
                if entry.is_dir() and entry.name == ".harbor":
                    continue

                if not show_hidden and entry.name.startswith("."):
                    continue

                relative_path = f"{prefix}{entry.name}"
                if entry.is_dir():
                    files.extend(collect_files(entry, f"{relative_path}/"))
                else:
                    files.append(relative_path)
            return files

        files = collect_files(directory_path)
        if not files:
            return "Directory is empty."

        result = ["Files in directory:"]
        result.extend(f"  {file}" for file in files[:max_files])
        if len(files) > max_files:
            result.append(f"  ({len(files) - max_files} more files not shown)")
        return "\n".join(result)
    except PermissionError:
        return f"Permission denied accessing directory: {directory}"
    except Exception as error:  # noqa: BLE001
        return f"Error listing directory: {error}"
