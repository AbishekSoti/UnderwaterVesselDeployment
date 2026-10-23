# Underwater Vessel Deployment

Benchmarking hardware configurations for embedded machine learning,
starting with underwater acoustic vessel classification.

The project investigates feature extraction, model inference, end-to-end
latency, memory usage, throughput, power consumption and system design.

Planned targets include Raspberry Pi, ESP32, FPGA, Astra and combined
systems such as ESP32–FPGA, Astra–FPGA and ARM–FPGA.

## Structure

- `reference/` — Reference feature extraction and inference.
- `models/` — Model documentation and locally stored checkpoints.
- `configs/` — Shared configuration.
- `platforms/` — Reusable device-specific implementations.
- `systems/` — Integration of multiple processing devices.
- `tools/` — Export, conversion and deployment scripts.
- `tests/` — Numerical and functional verification.
- `benchmarks/` — Measurement scripts and selected results.
- `docs/` — Setup instructions and design notes.

## Status

Initial project structure. Existing implementations will be imported
and verified incrementally.

Datasets, local environments, private configuration and model binaries
are excluded from Git by default.
