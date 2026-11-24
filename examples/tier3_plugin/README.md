# Tier 3 Code Plugin Example

Tier 3 tasks allow custom Python `Solver` and `Scorer` classes for complex multi-turn or programmatic benchmarks.

## Security Boundary
- Loading Tier 3 tasks requires explicit `--trust-remote-code`.
- Any model-generated code execution MUST run inside `eval_studio.sandbox.docker` with network disabled and strict CPU/memory/PID/timeout limits—never in-process on the host.
