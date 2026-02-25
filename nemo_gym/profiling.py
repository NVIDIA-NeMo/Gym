from io import StringIO
from pathlib import Path
from typing import Optional

import yappi
from gprof2dot import main as gprof2dot_main
from pydantic import BaseModel
from pydot import graph_from_dot_file


class Profiler(BaseModel):
    name: str
    base_profile_dir: Path

    # Used to clean up and filter out unnecessary information in the yappi log
    required_str: Optional[str] = None

    def model_post_init(self, context):
        assert " " not in self.name, f"Spaces are not allowed in profiler name, but got `{repr(self.name)}`"
        return super().model_post_init(context)

    def start(self) -> None:
        yappi.set_clock_type("CPU")
        yappi.start()
        print(f"🔍 Enabled profiling for {self.name}")

    def stop(self) -> None:
        print(f"🛑 Stopping profiler for {self.name}. Check {self.base_profile_dir} for the metrics!")
        yappi.stop()
        self.dump()

    def dump(self) -> None:
        self.base_profile_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.base_profile_dir / f"{self.name}.log"
        callgrind_path = self.base_profile_dir / f"{self.name}.callgrind"
        callgrind_dotfile_path = self.base_profile_dir / f"{self.name}.dot"
        callgrind_graph_path = self.base_profile_dir / f"{self.name}.png"

        yappi.get_func_stats().save(callgrind_path, type="CALLGRIND")
        gprof2dot_main(argv=f"--format=callgrind --output={callgrind_dotfile_path} -e 5 -n 5 {callgrind_path}".split())

        (graph,) = graph_from_dot_file(callgrind_dotfile_path)
        graph.write_png(callgrind_graph_path)

        buffer = StringIO()
        yappi.get_func_stats().print_all(
            out=buffer,
            columns={
                0: ("name", 200),
                1: ("ncall", 10),
                2: ("tsub", 8),
                3: ("ttot", 8),
                4: ("tavg", 8),
            },
        )

        buffer.seek(0)
        res = ""
        past_header = False
        for line in buffer:
            if not past_header or (self.required_str and self.required_str in line):
                res += line

            if line.startswith("name"):
                past_header = True

        with open(log_path, "w") as f:
            f.write(res)
