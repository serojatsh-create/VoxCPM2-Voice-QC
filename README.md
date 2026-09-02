# VoxCPM2 Voice Generation and QC Flow

This project generates Chinese voice segments with VoxCPM2, analyzes every candidate with ASR and acoustic checks, retries failed segments with deterministic seeds, freezes the selected source candidates, and performs a final whole-audio quality check.

## Workflow

1. Prepare corrected source text.
2. Segment it by meaning with the included `SEGMENTATION_PROMPT.md` using the AI tool you choose (Codex, Claude Code, or another model), then save one approved segment per line in `input.txt`.
3. Run the Python flow. It validates that the segment boundaries changed but the text did not.
4. VoxCPM2 generates each segment. Failed candidates are retried at most twice.
5. ASR, timing, speed, clipping, spectral, and format checks select the best passing candidate.
6. FFmpeg finds continuous silence. Intervals at or below 200 ms remain unchanged; longer intervals are edited only in a derivative copy and reduced to 200 ms.
7. The derivative segments are assembled and the whole result is checked again. `final.wav` exists only after the final check passes.

Original candidates remain frozen under `candidates/`; post-processing never overwrites them.

## Local setup

Copy `config.example.json` to `config.json` and replace the local tool and model paths. Keep `config.json`, model files, reference audio, and generated outputs out of GitHub. The current production configuration is machine-specific by design; the example configuration is the portable template.

Run a read-only environment check:

```powershell
& "C:\\path\\to\\VoxCPM2\\.venv\\Scripts\\python.exe" voice_qc_flow.py --config config.json --check-only
```

Run generation:

```powershell
& "C:\\path\\to\\VoxCPM2\\.venv\\Scripts\\python.exe" voice_qc_flow.py --config config.json
```

The ASR worker uses the separate ASR Python environment configured in `config.json`. No Prefect server or database is required by the current production entry point.

The generated Codex plugin is under `plugin/voxcpm2-voice-qc/`. It wraps the local entry point only; it does not contain model weights or private data.

## Licensing and third-party software

Add the license that applies to your own code before making this repository public. Keep the upstream licenses and notices for VoxCPM2, FunASR, FFmpeg, and any other dependency used by your installation. Model weights, reference recordings, datasets, and private text are not included in this repository.
