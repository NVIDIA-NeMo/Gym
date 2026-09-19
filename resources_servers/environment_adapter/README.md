# Environment adapter

This internal Resources Server is the default execution adapter for trusted authored environments.

It is intended to load tasks from `environment.yaml`, own a fresh sandbox and declared MCP processes for each episode, invoke the selected environment-local verifier, and clean up the episode.
The current implementation supports the singleton `hello-world` slice; tasksets, directory-authored tasks, and MCP process management remain to be implemented.

Environment authors use the files in their environment directory rather than configuring this server or writing a replacement.
A custom Resources Server is only needed for behavior outside the standard environment lifecycle, such as externally managed durable services, specialized seeding, or authoritative state that the adapter cannot own.
