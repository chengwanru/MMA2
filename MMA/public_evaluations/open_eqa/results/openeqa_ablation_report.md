# OpenEQA HM3D Ablation Report

**Benchmark:** [OpenEQA](https://open-eqa.github.io/) (Majumdar et al., CVPR 2024) episodic-memory split, HM3D scenes -- 557 open-vocabulary questions total (object recognition, spatial reasoning, object states, functional reasoning, world knowledge), each grounded in a short RGB frame sequence sampled per episode. Each group below runs the full 557-question set unless noted.

**Scoring methodology:** two metrics per group.
1. *Substring-match accuracy* (`good`) -- our own lightweight heuristic: EXACT/HIT/PARTIAL count as good, MISS/ERROR do not. Fast, free, used for iteration.
2. *LLM-Match* -- the official OpenEQA protocol: an LLM judge rates each prediction against the gold answer on a 1-5 scale, each score is linearly rescaled to 0-100 via `(score-1)/4*100`, and the group's number is the mean over all judged questions. ERROR/empty predictions are exported as an empty string (never a fabricated answer) before judging, so they score at or near the 1-5 floor on their own merits rather than being hand-corrected.

**Judge for this report:** deepseek / deepseek-v4-pro (base_url=https://ws-b6qf11bwei7je3nk.cn-beijing.maas.aliyuncs.com/compatible-mode/v1)

## Direct 8B (64f, no resize)
- Settings: frames=64, resize=N, hybrid=--, visual_routing=--, verify_reject_bad_draft=--, reject_strategy=--
- Paths:
  - Run directory: `/workspace/open_eqa_runs/direct_hm3d557_f64_20260727_231905`
  - Results JSON (predictions, variant=`baseline`): `/workspace/open_eqa_runs/direct_hm3d557_f64_20260727_231905/direct_vl8b.json`
  - `run.log`: `/workspace/open_eqa_runs/direct_hm3d557_f64_20260727_231905/run.log`
  - Per-question debug files: `/workspace/open_eqa_runs/direct_hm3d557_f64_20260727_231905/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/direct_hm3d557_f64_20260727_231905-baseline-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=104 hit=90 partial=62 miss=301 error=0 good=256/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.952/5) normalized=48.8/100 n_judged=557/557 distribution={'1': 204, '2': 27, '3': 101, '4': 42, '5': 183}

**Timing (per question, seconds)**

- Single VL forward pass (no MMA memory, no speculative decoding): mean=46.146s median=46.146s sum=25703.123s (n=557)


## Direct 8B (64f, resize)
- Settings: frames=64, resize=Y, hybrid=--, visual_routing=--, verify_reject_bad_draft=--, reject_strategy=--
- Paths:
  - Run directory: `/workspace/open_eqa_runs/direct_hm3d557_f64_resize_20260728_181653`
  - Results JSON (predictions, variant=`baseline`): `/workspace/open_eqa_runs/direct_hm3d557_f64_resize_20260728_181653/direct_vl8b.json`
  - `run.log`: `/workspace/open_eqa_runs/direct_hm3d557_f64_resize_20260728_181653/run.log`
  - Per-question debug files: `/workspace/open_eqa_runs/direct_hm3d557_f64_resize_20260728_181653/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/direct_hm3d557_f64_resize_20260728_181653-baseline-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=102 hit=85 partial=53 miss=317 error=0 good=240/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.939/5) normalized=48.5/100 n_judged=557/557 distribution={'1': 209, '2': 31, '3': 94, '4': 31, '5': 192}

**Timing (per question, seconds)**

- Single VL forward pass (no MMA memory, no speculative decoding): mean=0.873s median=0.789s sum=486.092s (n=557)


## Direct 8B (16f, no resize)
- Settings: frames=16, resize=N, hybrid=--, visual_routing=--, verify_reject_bad_draft=--, reject_strategy=--
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/direct_vl8b_novr_full557_f16_20260801_090636`
  - Results JSON (predictions, variant=`baseline`): `/nix/openeqa_work/open_eqa_runs/direct_vl8b_novr_full557_f16_20260801_090636/direct_vl8b.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/direct_vl8b_novr_full557_f16_20260801_090636/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/direct_vl8b_novr_full557_f16_20260801_090636/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/direct_vl8b_novr_full557_f16_20260801_090636-baseline-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=99 hit=73 partial=48 miss=337 error=0 good=220/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.792/5) normalized=44.8/100 n_judged=557/557 distribution={'1': 235, '2': 23, '3': 90, '4': 41, '5': 168}

**Timing (per question, seconds)**

- Single VL forward pass (no MMA memory, no speculative decoding): mean=0.298s median=0.24s sum=166.248s (n=557)


## MMA+SD (16f, no resize, no hybrid)
- Settings: frames=16, resize=N, hybrid=N, visual_routing=N, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f16_20260731_230701`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f16_20260731_230701/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f16_20260731_230701/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f16_20260731_230701/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_novr_full557_f16_20260731_230701-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=96 hit=80 partial=62 miss=294 error=25 good=238/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.641/5) normalized=41.0/100 n_judged=557/557 distribution={'1': 271, '2': 25, '3': 66, '4': 23, '5': 172}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=36.003s median=33.76s sum=20053.771s (n=557)
- Episodic memory write ('memorize') phase: mean=0.852s median=0.0s sum=474.72s (n=557)
- QA phase wall time: mean=35.151s median=33.089s sum=19579.051s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=1.926s median=1.387s sum=985.956s (n=512)
- Draft model (2B) token generation: mean=1.234s median=0.907s sum=631.916s (n=512)
- Target model (8B) forward pass: mean=0.655s median=0.399s sum=335.25s (n=512)
- Verification (accept/reject draft tokens): mean=0.037s median=0.017s sum=18.79s (n=512)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=512)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.006s (n=512)


## MMA+SD baseline (64f, no resize, no hybrid)
- Settings: frames=64, resize=N, hybrid=N, visual_routing=N, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f64_20260731_152721`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f64_20260731_152721/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f64_20260731_152721/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f64_20260731_152721/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_novr_full557_f64_20260731_152721-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=96 hit=87 partial=62 miss=304 error=8 good=245/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.704/5) normalized=42.6/100 n_judged=557/557 distribution={'1': 262, '2': 23, '3': 72, '4': 18, '5': 182}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=39.376s median=37.045s sum=21932.648s (n=557)
- Episodic memory write ('memorize') phase: mean=1.695s median=0.0s sum=944.303s (n=557)
- QA phase wall time: mean=37.681s median=36.181s sum=20988.345s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=1.834s median=1.439s sum=986.565s (n=538)
- Draft model (2B) token generation: mean=1.043s median=0.853s sum=561.318s (n=538)
- Target model (8B) forward pass: mean=0.755s median=0.483s sum=406.057s (n=538)
- Verification (accept/reject draft tokens): mean=0.036s median=0.017s sum=19.191s (n=538)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=538)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.006s (n=538)


## MMA+SD + visual routing (64f)
- Settings: frames=64, resize=N, hybrid=N, visual_routing=Y, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_visualrouting_full557_f64_20260731_160310`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_visualrouting_full557_f64_20260731_160310/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_visualrouting_full557_f64_20260731_160310/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_visualrouting_full557_f64_20260731_160310/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_visualrouting_full557_f64_20260731_160310-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=97 hit=87 partial=61 miss=304 error=8 good=245/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.722/5) normalized=43.0/100 n_judged=557/557 distribution={'1': 259, '2': 17, '3': 79, '4': 24, '5': 178}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=39.587s median=37.818s sum=22050.039s (n=557)
- Episodic memory write ('memorize') phase: mean=1.702s median=0.0s sum=948.227s (n=557)
- QA phase wall time: mean=37.885s median=36.669s sum=21101.812s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=1.322s median=0.905s sum=711.017s (n=538)
- Draft model (2B) token generation: mean=0.71s median=0.321s sum=381.832s (n=538)
- Target model (8B) forward pass: mean=0.588s median=0.477s sum=316.176s (n=538)
- Verification (accept/reject draft tokens): mean=0.024s median=0.017s sum=13.01s (n=538)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=538)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.007s (n=538)


## MMA+SD (64f, resize)
- Settings: frames=64, resize=Y, hybrid=N, visual_routing=N, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_resize_plain_full557_f64_20260731_230114`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_resize_plain_full557_f64_20260731_230114/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_resize_plain_full557_f64_20260731_230114/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_resize_plain_full557_f64_20260731_230114/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_resize_plain_full557_f64_20260731_230114-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=105 hit=71 partial=60 miss=300 error=21 good=236/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.689/5) normalized=42.2/100 n_judged=557/557 distribution={'1': 267, '2': 24, '3': 59, '4': 29, '5': 178}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=87.171s median=33.082s sum=48553.969s (n=557)
- Episodic memory write ('memorize') phase: mean=52.516s median=0.0s sum=29251.169s (n=557)
- QA phase wall time: mean=34.655s median=32.495s sum=19302.8s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=1.824s median=1.358s sum=941.386s (n=516)
- Draft model (2B) token generation: mean=1.047s median=0.828s sum=540.391s (n=516)
- Target model (8B) forward pass: mean=0.745s median=0.478s sum=384.537s (n=516)
- Verification (accept/reject draft tokens): mean=0.032s median=0.016s sum=16.46s (n=516)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=516)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.006s (n=516)


## MMA+SD + visual routing + block_verify (64f)
- Settings: frames=64, resize=N, hybrid=N, visual_routing=Y, verify_reject_bad_draft=N, reject_strategy=block_verify+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_blockverify_full557_f64_20260801_092252`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_blockverify_full557_f64_20260801_092252/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_blockverify_full557_f64_20260801_092252/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_blockverify_full557_f64_20260801_092252/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_vr_blockverify_full557_f64_20260801_092252-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=97 hit=87 partial=61 miss=304 error=8 good=245/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.74/5) normalized=43.5/100 n_judged=557/557 distribution={'1': 256, '2': 23, '3': 70, '4': 26, '5': 182}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=33.008s median=30.827s sum=18385.306s (n=557)
- Episodic memory write ('memorize') phase: mean=1.605s median=0.0s sum=894.091s (n=557)
- QA phase wall time: mean=31.403s median=30.406s sum=17491.215s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=1.356s median=0.913s sum=729.371s (n=538)
- Draft model (2B) token generation: mean=0.692s median=0.312s sum=372.475s (n=538)
- Target model (8B) forward pass: mean=0.602s median=0.479s sum=323.779s (n=538)
- Verification (accept/reject draft tokens): mean=0.062s median=0.054s sum=33.116s (n=538)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=538)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.007s (n=538)


## MMA+SD + hybrid_vl (64f)
- Settings: frames=64, resize=N, hybrid=Y, visual_routing=N, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_hybrid_full557_f64_20260731_152721`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_hybrid_full557_f64_20260731_152721/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_hybrid_full557_f64_20260731_152721/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_novr_hybrid_full557_f64_20260731_152721/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_novr_hybrid_full557_f64_20260731_152721-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=96 hit=99 partial=68 miss=289 error=5 good=263/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.901/5) normalized=47.5/100 n_judged=557/557 distribution={'1': 223, '2': 22, '3': 90, '4': 31, '5': 191}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=44.161s median=35.98s sum=24597.599s (n=557)
- Episodic memory write ('memorize') phase: mean=1.679s median=0.0s sum=935.134s (n=557)
- QA phase wall time: mean=42.482s median=34.009s sum=23662.465s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=1.056s median=0.784s sum=202.778s (n=192)
- Draft model (2B) token generation: mean=0.488s median=0.21s sum=93.69s (n=192)
- Target model (8B) forward pass: mean=0.524s median=0.467s sum=100.554s (n=192)
- Verification (accept/reject draft tokens): mean=0.044s median=0.023s sum=8.535s (n=192)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=192)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.002s (n=192)


## MMA+SD + attach-3-frames (64f)
- Settings: frames=64, resize=N, hybrid=N, visual_routing=N, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_attachframes3_full557_f64_20260731_225508`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_attachframes3_full557_f64_20260731_225508/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_attachframes3_full557_f64_20260731_225508/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_attachframes3_full557_f64_20260731_225508/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_attachframes3_full557_f64_20260731_225508-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=101 hit=96 partial=67 miss=292 error=1 good=264/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.873/5) normalized=46.8/100 n_judged=557/557 distribution={'1': 239, '2': 18, '3': 75, '4': 25, '5': 200}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=40.429s median=38.025s sum=22519.141s (n=557)
- Episodic memory write ('memorize') phase: mean=1.659s median=0.0s sum=924.144s (n=557)
- QA phase wall time: mean=38.77s median=37.523s sum=21594.997s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=5.702s median=5.27s sum=3152.953s (n=553)
- Draft model (2B) token generation: mean=1.835s median=1.672s sum=1014.761s (n=553)
- Target model (8B) forward pass: mean=3.835s median=2.982s sum=2120.695s (n=553)
- Verification (accept/reject draft tokens): mean=0.032s median=0.017s sum=17.496s (n=553)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=553)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.008s (n=553)


## MMA+SD + visual routing (64f, resize)
- Settings: frames=64, resize=Y, hybrid=N, visual_routing=Y, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_resize_full557_f64_20260801_193533`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_resize_full557_f64_20260801_193533/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_resize_full557_f64_20260801_193533/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_resize_full557_f64_20260801_193533/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_vr_resize_full557_f64_20260801_193533-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=105 hit=71 partial=60 miss=300 error=21 good=236/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.688/5) normalized=42.2/100 n_judged=557/557 distribution={'1': 264, '2': 28, '3': 62, '4': 24, '5': 179}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=40.455s median=37.509s sum=22533.467s (n=557)
- Episodic memory write ('memorize') phase: mean=1.718s median=0.0s sum=957.067s (n=557)
- QA phase wall time: mean=38.737s median=36.391s sum=21576.4s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=1.339s median=0.877s sum=691.139s (n=516)
- Draft model (2B) token generation: mean=0.742s median=0.3s sum=383.061s (n=516)
- Target model (8B) forward pass: mean=0.576s median=0.473s sum=297.192s (n=516)
- Verification (accept/reject draft tokens): mean=0.021s median=0.016s sum=10.885s (n=516)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=516)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.007s (n=516)


## MMA+SD + visual routing (16f)
- Settings: frames=16, resize=N, hybrid=N, visual_routing=Y, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_full557_f16_20260801_193533`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_full557_f16_20260801_193533/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_full557_f16_20260801_193533/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_full557_f16_20260801_193533/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_vr_full557_f16_20260801_193533-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=96 hit=81 partial=61 miss=294 error=25 good=238/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.619/5) normalized=40.5/100 n_judged=557/557 distribution={'1': 280, '2': 20, '3': 62, '4': 22, '5': 173}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=39.774s median=37.502s sum=22154.314s (n=557)
- Episodic memory write ('memorize') phase: mean=0.862s median=0.0s sum=480.358s (n=557)
- QA phase wall time: mean=38.912s median=36.585s sum=21673.956s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=1.398s median=0.845s sum=716.023s (n=512)
- Draft model (2B) token generation: mean=0.878s median=0.317s sum=449.467s (n=512)
- Target model (8B) forward pass: mean=0.496s median=0.401s sum=254.168s (n=512)
- Verification (accept/reject draft tokens): mean=0.024s median=0.017s sum=12.387s (n=512)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=512)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.006s (n=512)


## MMA+SD + visual routing + attach-3-frames (64f)
- Settings: frames=64, resize=N, hybrid=N, visual_routing=Y, verify_reject_bad_draft=N, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_attachframes3_full557_f64_20260801_193533`
  - Results JSON (predictions, variant=`ours`): `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_attachframes3_full557_f64_20260801_193533/mma_sd.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_attachframes3_full557_f64_20260801_193533/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/mma_sd_vr_attachframes3_full557_f64_20260801_193533/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/mma_sd_vr_attachframes3_full557_f64_20260801_193533-ours-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=103 hit=97 partial=66 miss=290 error=1 good=266/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.862/5) normalized=46.6/100 n_judged=557/557 distribution={'1': 245, '2': 15, '3': 68, '4': 30, '5': 199}

**Timing (per question, seconds)**

- Total wall time (memorize + QA): mean=43.357s median=41.012s sum=24149.993s (n=557)
- Episodic memory write ('memorize') phase: mean=1.696s median=0.0s sum=944.411s (n=557)
- QA phase wall time: mean=41.662s median=40.352s sum=23205.582s (n=557)
- Speculative decoding total (draft+align+target+verify, summed across rounds): mean=5.241s median=4.908s sum=2882.754s (n=550)
- Draft model (2B) token generation: mean=1.541s median=1.333s sum=847.605s (n=550)
- Target model (8B) forward pass: mean=3.678s median=2.999s sum=2022.701s (n=550)
- Verification (accept/reject draft tokens): mean=0.023s median=0.017s sum=12.448s (n=550)
- Memory RoPE realignment: mean=0.0s median=0.0s sum=0.0s (n=550)
- Unaccounted scheduling/alignment overhead (decode total minus the four components above): mean=0.0s median=0.0s sum=0.007s (n=550)


## Pure SD, no MMA (64f, resize)
- Settings: frames=64, resize=Y, hybrid=--, visual_routing=--, verify_reject_bad_draft=--, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/direct_sd_resize_hm3d557_f64_20260731_005416`
  - Results JSON (predictions, variant=`direct_sd`): `/nix/openeqa_work/open_eqa_runs/direct_sd_resize_hm3d557_f64_20260731_005416/direct_sd.raw.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/direct_sd_resize_hm3d557_f64_20260731_005416/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/direct_sd_resize_hm3d557_f64_20260731_005416/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/direct_sd_resize_hm3d557_f64_20260731_005416-direct_sd-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=86 hit=80 partial=75 miss=316 error=0 good=241/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.686/5) normalized=42.1/100 n_judged=557/557 distribution={'1': 240, '2': 30, '3': 106, '4': 27, '5': 154}

**Timing (per question, seconds)**

- Speculative decoding, no MMA memory (draft+target, run_openeqa_direct_sd.py): mean=1.746s median=1.427s sum=972.415s (n=557)


## Pure SD, no MMA (16f, no resize)
- Settings: frames=16, resize=N, hybrid=--, visual_routing=--, verify_reject_bad_draft=--, reject_strategy=greedy+semantic
- Paths:
  - Run directory: `/nix/openeqa_work/open_eqa_runs/direct_sd_novr_full557_f16_20260802_130351`
  - Results JSON (predictions, variant=`compare_sd`): `/nix/openeqa_work/open_eqa_runs/direct_sd_novr_full557_f16_20260802_130351/direct_sd.raw.json`
  - `run.log`: `/nix/openeqa_work/open_eqa_runs/direct_sd_novr_full557_f16_20260802_130351/run.log`
  - Per-question debug files: `/nix/openeqa_work/open_eqa_runs/direct_sd_novr_full557_f16_20260802_130351/openeqa_debug/`
  - LLM-Match per-question scores (resumable cache): `/tmp/MMA2/MMA/public_evaluations/open_eqa/results/metrics/direct_sd_novr_full557_f16_20260802_130351-compare_sd-deepseek-deepseek-v4-pro-metrics.json`

**Substring-match accuracy**

total=557 exact=86 hit=77 partial=65 miss=329 error=0 good=228/557

**LLM-Judge score** (judge model: `deepseek-v4-pro`)

mean=(2.596/5) normalized=39.9/100 n_judged=557/557 distribution={'1': 260, '2': 29, '3': 93, '4': 26, '5': 149}

**Timing (per question, seconds)**

- Speculative decoding, no MMA memory (draft+target, run_openeqa_direct_sd.py): mean=0.508s median=0.444s sum=282.992s (n=557)

