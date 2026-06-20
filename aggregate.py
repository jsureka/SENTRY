import json
import os

RESULTS_DIR = 'results'
OUT_PATH = os.path.join('writeup', 'Aggregated_Raw_Results.md')

names = [
    'CI_kNN_defect_codebert_results',
    'CI_kNN_defect_graphcodebert_results',
    'CI_kNN_vuln_codebert_results',
    'CI_kNN_vuln_graphcodebert_results'
]
dirs = [os.path.join(RESULTS_DIR, n) for n in names]

out = "# Aggregated Experimental Results\n\n"

for d in dirs:
    out += f"## Experiment: {os.path.basename(d)}\n\n"

    # 1. Final Table
    suffix = os.path.basename(d).replace('CI_kNN_','').replace('_results','')
    p1 = os.path.join(d, f'final_table_{suffix}.json')
    if os.path.exists(p1):
        out += "### Final Pipeline Metrics\n"
        out += "```json\n" + json.dumps(json.load(open(p1)), indent=2) + "\n```\n\n"
        
    # 2. Ablation Results
    p2 = os.path.join(d, 'ablation_results.json')
    if os.path.exists(p2):
        out += "### k/λ Parameter Ablation Results\n"
        out += "```json\n" + json.dumps(json.load(open(p2)), indent=2) + "\n```\n\n"
        
    # 3. OOD Detection
    p3 = os.path.join(d, 'ood_detection_results.json')
    if os.path.exists(p3):
        out += "### OOD Detection Metrics (Mahalanobis, Energy, RMD, Baseline)\n"
        out += "```json\n" + json.dumps(json.load(open(p3)), indent=2) + "\n```\n\n"
        
    # 4. Conformal Prediction
    p4 = os.path.join(d, 'conformal', 'conformal_metrics.json')
    if os.path.exists(p4):
        out += "### Conformal Prediction Results (RAPS/LAC)\n"
        out += "```json\n" + json.dumps(json.load(open(p4)), indent=2) + "\n```\n\n"
        
    # 5. McNemar Test
    p5 = os.path.join(d, 'significance', 'mcnemar_results.json')
    if os.path.exists(p5):
        out += "### McNemar Statistical Significance Tests\n"
        out += "```json\n" + json.dumps(json.load(open(p5)), indent=2) + "\n```\n\n"

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(out)
print(f"Aggregation complete -> {OUT_PATH}")
