# Direct SD @ 64 frames failure notes (HM3D 557 attempt)

**What this is:** Pure speculative decoding on OpenEQA frames (**no MMA memory**),
`run_openeqa_direct_sd.py --mode speculative --max_frames 64`.

**Not this:** `direct_hm3d557_f64_*` runs are Direct **8B baseline** (`--mode baseline`).

## Why it failed (summary)

With 64 images in the VL prompt, each SD round runs draft **and** target over the full
visual prefix:

- Draft ~19s / round
- Target ~45s / round
- Often `acceptance_rate=0` and empty `pred=''`
- Per-question wall ~60–190s → not viable for full 557; typically **slower** than Direct 8B

So this is a **setting mismatch** (multi-frame VL + SD without memory compression), not a missing script.

## Files

- `run_excerpt.log` — representative lines from the AIBox run (profile + empty/slow samples)
- Full AIBox dirs (if still present): `/workspace/open_eqa_runs/direct_sd_hm3d557_f64_*/`

## Related paper baselines

| Arm | Dir prefix on AIBox |
|-----|---------------------|
| Direct 8B | `direct_hm3d557_f64_*` |
| Pure SD | `direct_sd_hm3d557_f64_*` |
| MMA±Hybrid | `mma_sd_*` / `mma_hybrid_*` |
