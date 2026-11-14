# PDF Analyst QA

This environment trains agents to answer questions about PDFs. It uses Gym's Environment Server and OpenCode session harness. The resources server selects the PDFs, manages the sandbox, and keeps reference answers separate from agent input. Answers receive 1 when they are correct and follow the requested format, and 0 otherwise.

Supply a dataset YAML that includes [config.yaml](config.yaml) and declares the dataset paths and license. [example.yaml](example.yaml) uses the included example dataset and serves as a template. [Dataset and runtime setup](../../resources_servers/pdf_analyst_qa/README.md) describes the expected files and how to run the environment.
