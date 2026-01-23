from io import StringIO
from pathlib import Path

import yappi
from gprof2dot import main as gprof2dot_main
from pydantic import BaseModel
from pydot import graph_from_dot_file


class Profiler(BaseModel):
    name: str
    base_profile_dir: Path

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
        log_path = self.base_profile_dir / "yappi.log"
        callgrind_path = self.base_profile_dir / "yappi.callgrind"
        callgrind_dotfile_path = self.base_profile_dir / "yappi.dot"
        callgrind_graph_path = self.base_profile_dir / "yappi.png"

        yappi.get_func_stats().save(callgrind_path, type="CALLGRIND")
        gprof2dot_main(argv=f"--format=callgrind --output={callgrind_dotfile_path} -e 1 -n 1 {callgrind_path}".split())

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
            if not past_header or self.config.entrypoint in line:
                res += line

            if line.startswith("name"):
                past_header = True

        with open(log_path, "w") as f:
            f.write(res)
