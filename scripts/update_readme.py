import re
from pathlib import Path


README_PATH = Path("README.md")

TARGET_FOLDER = Path("resources_servers")


def generate_table():
    header = "|Server Type Name|Domain|Path|\n|----|----|---|"

    rows = []
    for subdir in sorted(TARGET_FOLDER.iterdir()):
        path = f"`{TARGET_FOLDER.name}/{subdir.name}`"
        pretty = subdir.name.replace("_", " ").title()
        placeholder = "?"  # TODO: find out where domain lives
        rows.append(f"|{pretty}|{placeholder}|{path}|")

    return "\n".join([header] + rows)


def main():
    text = README_PATH.read_text()
    pattern = re.compile(
        r"(<!-- START_RESOURCE_TABLE -->)(.*?)(<!-- END_RESOURCE_TABLE -->)",
        flags=re.DOTALL,
    )
    new_text = pattern.sub(lambda m: f"{m.group(1)}\n{generate_table()}\n{m.group(3)}", text)
    README_PATH.write_text(new_text)
    # print(new_text)


if __name__ == "__main__":
    main()
