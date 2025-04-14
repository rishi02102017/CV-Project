import os
import json
from utils.metrics import compute_metrics
from datasets.lane_dataset import LaneDataset
from utils.tokenizer import Tokenizer
import yaml
from tqdm import tqdm

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Prepare paths
gt_dir = config['evaluation']['ground_truth_dir']
pred_dir = config['evaluation']['prediction_dir']

# Initialize tokenizer
tokenizer = Tokenizer(config)

# Load GT and Predictions
gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]

assert len(gt_files) == len(pred_files), "Mismatch in number of GT and prediction files."

all_metrics = []

for file_name in tqdm(gt_files, desc="Evaluating"):
    gt_path = os.path.join(gt_dir, file_name)
    pred_path = os.path.join(pred_dir, file_name)

    with open(gt_path, 'r') as f:
        gt = json.load(f)

    with open(pred_path, 'r') as f:
        pred = json.load(f)

    # Compute metrics for the sample
    metrics = compute_metrics(gt, pred, config)
    all_metrics.append(metrics)

# Aggregate metrics
precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
f1_score = 2 * precision * recall / (precision + recall + 1e-6)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}")
