---
name: voice-qc
description: Run the local VoxCPM2 semantic segmentation, GPU-protected generation, ASR quality checks, and final audio validation workflow.
---

# VoxCPM2 Voice QC

Use this skill for the local VoxCPM2 voice-generation and quality-control workflow.

## Required sequence

1. Read the complete source text.
2. Perform semantic segmentation with the AI currently executing the task. Do not require Ollama, OpenAI API, Claude API, or another fixed provider.
3. Preserve every source character. Only boundaries may change.
4. Treat 120-180 characters as a soft reference, never as a mechanical target. Keep a shorter or longer segment when its meaning and spoken delivery require it.
5. Never allow a segment over 240 characters into generation.
6. Run `--check-only` with the configured VoxCPM2 Python interpreter.
7. Confirm GPU free memory is at least 4096 MB and no workflow run is active.
8. Generate serially. Do not parallelize VoxCPM2 segments.
9. Inspect the report before claiming success. `final.wav` is valid only after final whole-audio QC passes.

## Entry points

Project root is three levels above this plugin directory. The local launcher is `scripts/run_voice_qc.ps1`.

Never include `config.json`, model paths, credentials, private text, reference audio, or output files in a response or commit.
