# OpenEQA sample logs for review

## Prefer these (early 50Q, actually worked)
- `direct_8b_n50/direct_o150_n50_f64_*` — Direct 8B @64, offset150, 50Q
- `mma_sd_n50/compare_o150_n50_f64_*` — MMA+SD @64, same slice (good≈26/50)
- `mma_sd_n50/compare_o100_n50_f16_*` — Direct vs MMA @16, offset100 (direct 23/50, mma 19/50)

## Note
557 Direct under old `direct_8b/` were error runs (empty data / hub version) and should be removed.
Current 557 MMA±Hybrid runs may still be under `mma_sd_nohybrid/` / `mma_hybrid/` if kept.
