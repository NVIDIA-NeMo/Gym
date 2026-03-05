# Description

Data links: ?

# Session affinity (multiple workers)

Sessions are stored in-memory per process. For multiple workers you must use **client-side session affinity** so all requests for a session hit the same worker:

1. Run each worker on a different port (e.g. `http://host:8001`, `http://host:8002`).
2. In the resources server config, set `worker_urls` to the list of worker base URLs:
   ```yaml
   worker_urls:
     - "http://host:8001"
     - "http://host:8002"
   ```
3. The GDPVal agent passes `affinity_key=session_id` on every resources server call; `ServerClient` hashes the key to choose the URL. All calls for the same session then go to the same worker.

If you use a single worker (`num_workers: 1`), you can omit `worker_urls`.

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
