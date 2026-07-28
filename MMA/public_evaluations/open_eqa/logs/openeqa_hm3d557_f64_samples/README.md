# OpenEQA HM3D≈557 @64 frames — sample run logs (AIBox)

- `direct_8b/` — Direct Qwen3-VL-8B (no MMA, no SD)
- `mma_sd_nohybrid/` — MMA + SD, no VR, greedy+semantic, no Hybrid VL
- `mma_hybrid/` — same + OPENEQA_QA_HYBRID_VL=1 (k=2)

Logs are copied as `run.txt` because `*.log` is gitignored.
Source machine paths under `/workspace/open_eqa_runs/`.
