"""BGL alert categories as exact signatures: (category, component, levels, regex on the message). Derived from the
LogHub BGL labels (Oliner and Stearley 2007); _build/bgl_verify.py checks them against every labeled line."""
import re

RULES = [
    ("KERNDTLB", "KERNEL", "FATAL", r"^data TLB error interrupt"),
    ("KERNTLBE", "KERNEL", "FATAL", r"^instruction TLB error interrupt"),
    ("KERNSTOR", "KERNEL", "FATAL", r"^data storage interrupt"),
    ("KERNMNTF", "KERNEL", "FATAL", r"^Lustre mount FAILED"),
    ("KERNMNT", "KERNEL", "FATAL", r"^Error: unable to mount filesystem"),
    ("KERNTERM", "KERNEL", "FATAL", r"^rts: kernel terminated for reason"),
    ("KERNRTSP", "KERNEL", "FATAL", r"^rts panic! - stopping execution"),
    ("KERNRTSA", "KERNEL", "FATAL", r"^rts assertion failed"),
    ("KERNREC", "KERNEL", "FATAL", r"^Error receiving packet on tree network"),
    ("KERNMICRO", "KERNEL", "FATAL", r"^Microloader Assertion"),
    ("KERNMC", "KERNEL", "FATAL", r"^(machine check interrupt|DDR machine check register)"),
    ("KERNSOCK", "KERNEL", "FATAL", r"socket closed$"),
    ("KERNCON", "KERNEL", "FATAL", r"BGLERR_IDO_PKT_TIMEOUT connection lost"),
    ("KERNPOW", "KERNEL", "FATAL", r"^Power deactivated:"),
    ("KERNSERV", "KERNEL", "FATAL", r"^Power Good signal deactivated:"),
    ("KERNPAN", "KERNEL", "FATAL", r"^kernel panic$"),
    ("KERNNOETH", "KERNEL", "FATAL", r"^no ethernet link$"),
    ("KERNPROG", "KERNEL", "FATAL", r"^program interrupt$"),
    ("KERNFLOAT", "KERNEL", "FATAL", r"^floating point unavailable interrupt"),
    ("KERNEXT", "KERNEL", "FATAL", r"tree header with no target waiting"),
    ("KERNBIT", "KERNEL", "FATAL", r"redundant bit steering failed"),
    ("APPTORUS", "APP", "FATAL", r"uncorrectable torus error"),
    ("APPSEV", "APP", "FATAL", r"^ciod: Error reading message prefix.*Link has been severed"),
    ("APPRES", "APP", "FATAL", r"^ciod: Error reading message prefix.*Connection reset by peer"),
    ("APPTO", "APP", "FATAL", r"^ciod: Error reading message prefix.*Connection timed out"),
    ("APPREAD", "APP", "FATAL", r"^ciod: failed to read message prefix on control stream"),
    ("APPOUT", "APP", "FATAL", r"^ciod: LOGIN chdir\(.*\) failed: Input/output error"),
    ("APPUNAV", "APP", "FATAL", r"^ciod: Error creating node map from file .*: Resource temporarily unavailable"),
    ("APPBUSY", "APP", "FATAL", r"^ciod: Error creating node map from file .*: Device or resource busy"),
    ("APPCHILD", "APP", "FATAL", r"^ciod: Error creating node map from file .*: No child processes"),
    ("APPALLOC", "APP", "FATAL", r"^ciod: Error creating node map from file .*: Cannot allocate memory"),
    ("LINKIAP", "LINKCARD", "FATAL", r"iap failed"),
    ("LINKDISC", "LINKCARD", "FATAL", r"port disconnected"),
    ("LINKPAP", "LINKCARD", "FATAL", r"pap failed"),
    ("LINKBLL", "LINKCARD", "FATAL", r"bll_clear_port failed"),
    ("MASABNORM", "BGLMASTER", "FAILURE", r"exited abnormally due to signal"),
    ("MASNORM", "BGLMASTER", "FAILURE", r"^mmcs_server exited normally"),
    ("MONPOW", "MONITOR", "FAILURE", r"power module"),
    ("MONNULL", "MONITOR", "FAILURE", r"NullPointerException"),
    ("MMCS", "MMCS", "FATAL", r"major internal error"),
]
_C = [(c, comp, lvl, re.compile(rx)) for c, comp, lvl, rx in RULES]


def split(raw):
    """A BGL line without its label column -> (component, level, message)."""
    p = raw.split(" ", 8)
    return (p[6], p[7], p[8]) if len(p) == 9 else ("", "", raw)


def categorize(raw):
    comp, lvl, msg = split(raw)
    for c, cc, ll, rx in _C:
        if comp == cc and lvl == ll and rx.search(msg):
            return c
    return None
