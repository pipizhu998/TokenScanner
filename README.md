# TokenScanner

**TokenScanner: Detecting Backdoors and Discovering Triggers in Text-to-Image LoRAs via Full Vocabulary Scanning**

[Paper](https://arxiv.org/abs/2609.31878)

This repository provides an implementation of **TokenScanner**, a model-level method for detecting backdoors and discovering potential trigger tokens in text-to-image LoRAs.

## Installation

Install the required dependencies:

```bash
pip install torch diffusers transformers peft accelerate safetensors numpy pandas pyyaml
```

## Usage

All experiments are configured through YAML files and executed using:

```bash
python main.py --config config.yaml
```

The pipeline consists of three steps.

### 1. Token Scanning

Scan a LoRA to compute token-level and model-level scores.

```yaml
task: scan
output_dir: outputs/reference

scan:
  adapter: /path/to/adapter.safetensors
  base_model: /path/to/stable-diffusion-v1-5
  name: benign_001
  label: benign
  device: cuda
```

### 2. Threshold Setting

Set detection thresholds using benign reference LoRAs.

```yaml
task: thresholding
output_dir: outputs

thresholding:
  input: outputs/reference/scans
  quantile: 0.9
```

### 3. Evaluation

Evaluate backdoor detection and trigger discovery.

```yaml
task: eval
output_dir: outputs

eval:
  input: outputs/test/scans
  thresholds: outputs/thresholds/thresholds.json
  ks: [1, 5]
```

## Configuration

The following YAML options are supported.

### General

| Option       | Default  | Description                       |
| ------------ | -------- | --------------------------------- |
| `task`       | Required | `scan`, `thresholding`, or `eval` |
| `output_dir` | `output` | Root directory for outputs        |

### Scan Options

Options under `scan:`:

| Option             | Default            | Description                                      |
| ------------------ | ------------------ | ------------------------------------------------ |
| `adapter`          | Required           | Path to the LoRA weights                         |
| `base_model`       | Required           | Path to the base Stable Diffusion model          |
| `name`             | Adapter filename   | Unique model identifier                          |
| `label`            | `null`             | `benign` or `backdoored`                         |
| `family`           | `null`             | Optional model category or attack family         |
| `trigger`          | `null`             | Ground-truth trigger for evaluation              |
| `device`           | `cuda`             | Computation device (`cuda`, `cuda:0`, or `cpu`)  |
| `unet_only`        | `false`            | Disable text encoder scanning                    |
| `te_only`          | `false`            | Disable U-Net scanning                           |
| `templates`        | Built-in templates | Prompt templates used for scanning               |
| `a`                | `5`                | Top-a parameter for U-Net concentration score    |
| `cache_batch_size` | `1024`             | Batch size for CLIP feature extraction           |
| `scan_batch_size`  | `4096`             | Batch size for U-Net response computation        |
| `te_batch_size`    | `8192`             | Batch size for text encoder response computation |

Default prompt templates:

```yaml
templates:
  - "a photo of {token}"
  - "a person with {token}"
  - "an object near {token}"
  - "a landscape containing {token}"
```

Each template must contain exactly one `{token}` placeholder.

### Threshold Setting Options

Options under `thresholding:`:

| Option     | Default  | Description                                 |
| ---------- | -------- | ------------------------------------------- |
| `input`    | Required | Directory containing benign reference scans |
| `quantile` | `0.9`    | Quantile used to set detection thresholds   |

### Evaluation Options

Options under `eval:`:

| Option       | Default                                 | Description                           |
| ------------ | --------------------------------------- | ------------------------------------- |
| `input`      | Required                                | Directory containing evaluation scans |
| `thresholds` | `output_dir/thresholds/thresholds.json` | Path to the threshold file            |
| `ks`         | `[1, 5]`                                | Top-K values for trigger discovery    |

## Outputs

| File                           | Description                            |
| ------------------------------ | -------------------------------------- |
| `token_scores.csv`             | Token-level U-Net/TE scores and ranks  |
| `token_scores_unet_ranked.csv` | Tokens ranked by U-Net score           |
| `token_scores_te_ranked.csv`   | Tokens ranked by text encoder score    |
| `model_scores.csv`             | Model-level scores                     |
| `metadata.json`                | Model metadata and trigger information |
| `thresholds.json`              | Detection thresholds                   |
| `decisions.csv`                | Backdoor detection results             |
| `hits.csv`                     | Trigger discovery results              |
| `metrics.json`                 | Evaluation metrics                     |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{liu2026tokenscanner,
  title={TokenScanner: Detecting Backdoors and Discovering
         Triggers in Text-to-Image LoRAs via
         Full Vocabulary Scanning},
  author={Boliang Liu and Jing Zhang},
  year={2026},
  eprint={2609.31878},
  archivePrefix={arXiv},
  primaryClass={cs.CR}
}
```
