# Recorded-context code and data provenance

The replay runtime is independently written NVIDIA Apache-2.0 code. Case prompts,
history, source snapshots, recorded responses and private provenance are external
data supplied by an authenticated asset package. The runner checks their declared
identities and does not import or execute those source snapshots.

Detailed internal data provenance and historical curation notes are retained in
the private data repository. Public fixtures are explicitly synthetic and contain
no customer artifacts. Production credentials and rollouts are excluded from git.

The existing advertised-tool checker and Prime child integration originated in
public NVIDIA-NeMo/Gym. Internal backport headers identify their exact public
source revisions and Apache-2.0 license. Retain required attribution when porting
adaptations; private session-API compatibility shims do not replace public APIs.
