# Mini LLM/AI API Gateway

Mini API Gateway exposes an OpenAI-compatible surface that can
route requests to multiple backend providers such as Ollama,
OpenAI, and Anthropic. It supports per-app policies, cost
accounting, scheduling, and configuration hot-reload via JSON files.

## Quick Start

Install dependencies and package:

```bash
pip install -e .
```

Populate the `config/` directory with `daemon.json`, `backends.json`, and `apps.json` (sample files are provided).

Run the gateway:

```bash
mini-apigw --config-dir ./config
```

Query the OpenAI-compatible API surface using your configured API key.

