# MMA Agent Framework

This directory contains the implementation of the **Multimodal Memory Agent (MMA)**, built upon the [MIRIX](https://github.com/Mirix-AI/MIRIX) architecture. It includes the core memory managers, the confidence module, and evaluation scripts for standard benchmarks (FEVER, LOCOMO).

## 📂 Code Structure

- **`MMA/`**: The core package directory (capital **MMA**). Code imports **`mma`** (lowercase). On **Linux** (case-sensitive disk), run once from this directory: `bash scripts/ensure_mma_import_symlink.sh` to create `mma -> MMA` if needed. On default **macOS** disks, names are case-insensitive so `import mma` resolves to `MMA/` without a separate symlink.
  - `services/confidence_module.py`: **[Core Contribution]** Implements Source, Time, and Consensus scoring.
  - `agent/meta_memory_agent.py`: Orchestrates memory retrieval and updates.
- **`public_evaluations/`**: Evaluation scripts.
  - `run_fever_eval.py`: For Fact Verification tasks.
  - `run_instance.py`: For Long-context QA tasks (LOCOMO).
- **`configs/`**: Configuration files (YAML).

## 🔧 Configuration Details

### Confidence Settings (`configs/confidence_v2.yaml`)

You can adjust the weights for the confidence score in the config file:

```yaml
confidence:
  w_s: 0.45 # Source Reliability Weight
  w_t: 0.40 # Temporal Decay Weight
  w_c: 0.15 # Network Consensus Weight
  time_half_life_days: 30
```

### Running Ablations

To run ablation studies (e.g., without Consensus), you can use the command line flags in `run_fever_eval.py`:

- `--formula_modes st`: Runs Source + Time (No Consensus).
- `--formula_modes tri`: Runs Full Model.

## 🚀 Advanced Usage

(If you have specific instructions on how to start the frontend or custom agents, add them here)
