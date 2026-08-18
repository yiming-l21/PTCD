# Configuration Files

`configs/paper/` contains the release configurations used by the PTCD paper reproduction scripts. These files keep the paper defaults explicit: 8 textual soft tokens, 16 visual prefix tokens, 1500 training steps, 200 warmup steps, batch size 4, and probability-level demo gating.

The dataset JSON files directly under `configs/` are kept as lightweight compatibility presets for older scripts and additional datasets. For the main paper results, prefer `configs/paper/<dataset>.json`.
