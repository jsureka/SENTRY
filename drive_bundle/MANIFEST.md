# SENTRY — Drive bundle manifest

Artifacts too large for git (regenerable), hosted on Google Drive for exact reproduction
without retraining. After uploading this folder to Drive, paste the shared link into the
README (`<!-- DRIVE_LINK -->` marker) and into `results/FINAL_VERDICT.md` if desired.

## Contents to upload (copy these into this folder before uploading)

| File | Size | Source / regenerate |
|---|---|---|
| `models/codebert_defect_model.bin`      | 476 MB | fine-tune via `Defect-Prediction/code/` |
| `models/codebert_vuln_model.bin`        | 476 MB | fine-tune via `Vulnerability-Detection/code/` |
| `models/graphcodebert_defect_model.bin` | 476 MB | fine-tune via `Defect-Prediction/code/` (GraphCodeBERT) |
| `models/graphcodebert_vuln_model.bin`   | 476 MB | fine-tune via `Vulnerability-Detection/code/` (GraphCodeBERT) |
| `datastores/CI_kNN_*_results/datastore/`| 4×65 MB| rebuild: `reproduce_results.py --rebuild_datastore` |
| `grid_emb/` (frozen-embedding caches)   | ~1.5 GB| rebuild: `run_grid.py` + `embed_clone.py` |

## SHA-256 (fill after staging, for integrity)
_run `shasum -a 256 models/*.bin` and paste here_

## SHA-256 (staged 2026-06-26)
```
03280eb89908eb3c21f8bb05f429c9331581eb132000d92f63bbecb428f1d4b9  models/codebert_defect_model.bin
b6315649f2d9681a1427fa9302a5714824eec24b69a292f40e6f83a4a5568249  models/codebert_vuln_model.bin
028fb72f4a371919c860db8b01a8504c4ed0d7529f9eb3acd5e70e0b9db43409  models/graphcodebert_defect_model.bin
0213aaa163a2a606de4a38928cf59ac43c95c667cb3f1873bfc4b67fc4b22673  models/graphcodebert_vuln_model.bin
```
