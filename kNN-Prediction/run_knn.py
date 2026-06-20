"""
run_knn.py — Main entry point for the kNN Retrieval-Augmented Prediction pipeline.

Usage:
    # Step 1: Prepare mini dataset
    python prepare_mini_dataset.py

    # Step 2: Build datastore + calibrate + evaluate (all at once)
    python run_knn.py \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --train_data ./mini_dataset/train.jsonl \
        --eval_data ./mini_dataset/dev.jsonl \
        --test_data ./mini_dataset/test.jsonl \
        --block_size 400 --num_labels 4 \
        --k 8 --lambda_val 0.3 --knn_temperature 10.0 \
        --embedding_strategy last_cls \
        --build_datastore --calibrate --evaluate

    # Or with a fine-tuned checkpoint:
    python run_knn.py \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --model_checkpoint ../Defect-Prediction/code/saved_models/checkpoint-best-acc/model.bin \
        ... (same as above)
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

# Add parent directory to path for model import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Defect-Prediction', 'code'))

from transformers import (
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    BertConfig, BertForSequenceClassification, BertTokenizer,
)

from knn_datastore import KNNDatastore, CodeDataset, EmbeddingExtractor
from knn_predictor import KNNPredictor
from calibration import TemperatureScaler, compute_entropy
from knn_evaluate import KNNEvaluator


MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}


class ModelWrapper(torch.nn.Module):
    """
    Wrapper that matches CodeImprove's Model class interface.
    Wraps a HuggingFace encoder and adds dropout.
    Provides both softmax probabilities AND raw logits.
    """
    def __init__(self, encoder, config, tokenizer, dropout_prob=0.0):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits = self.dropout(outputs[0] if isinstance(outputs, tuple) else outputs.logits)
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            return loss, prob
        return prob
    
    def forward_with_logits(self, input_ids):
        """Return both probabilities AND raw logits (for calibration)."""
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        prob = torch.softmax(logits, dim=-1)
        return prob, logits


def load_model(args):
    """Load model and tokenizer."""
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    config = config_class.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels
    
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model = ModelWrapper(model, config, tokenizer, dropout_prob=0.0)
    
    # Load fine-tuned checkpoint if provided
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        print(f"Loading fine-tuned checkpoint: {args.model_checkpoint}")
        state_dict = torch.load(args.model_checkpoint, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
        print("  Checkpoint loaded successfully")
    else:
        print("  Using pre-trained model (no fine-tuned checkpoint)")
    
    model.to(args.device)
    model.eval()
    
    return model, tokenizer


def extract_all_outputs(model, dataset, device, batch_size=32, strategy='last_cls'):
    """
    Run model on dataset and extract: probabilities, logits, embeddings, labels, IDs.
    """
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size
    )
    
    extractor = EmbeddingExtractor(model, device, strategy=strategy)
    
    all_probs = []
    all_logits = []
    all_embeddings = []
    all_labels = []
    all_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting outputs"):
            input_ids = batch[0].to(device)
            labels = batch[1].numpy()
            ids = batch[2]
            
            # Get probs and logits
            prob, logits = model.forward_with_logits(input_ids)
            all_probs.append(prob.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            
            # Get embeddings
            embeddings = extractor.extract(input_ids)
            all_embeddings.append(embeddings)
            
            all_labels.extend(labels.tolist())
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.numpy().tolist())
            else:
                all_ids.extend(list(ids))
    
    return {
        'probs': np.concatenate(all_probs, axis=0),
        'logits': np.concatenate(all_logits, axis=0),
        'embeddings': np.concatenate(all_embeddings, axis=0),
        'labels': np.array(all_labels),
        'ids': np.array(all_ids),
    }


def main():
    parser = argparse.ArgumentParser(description='kNN Retrieval-Augmented Prediction Pipeline')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='roberta', choices=['roberta', 'bert'])
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Path to fine-tuned model.bin checkpoint')
    parser.add_argument('--block_size', type=int, default=400)
    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--eval_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, required=True)
    
    # kNN arguments
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--lambda_val', type=float, default=0.3,
                        help='Interpolation weight (1.0=model only, 0.0=kNN only)')
    parser.add_argument('--knn_temperature', type=float, default=10.0)
    parser.add_argument('--voting', type=str, default='distance_weighted',
                        choices=['uniform', 'distance_weighted', 'threshold_filtered'])
    parser.add_argument('--embedding_strategy', type=str, default='last_cls',
                        choices=['last_cls', 'second_last_cls', 'avg_last4_cls', 'mean_pool_last'])
    
    # Pipeline control
    parser.add_argument('--build_datastore', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--run_all', action='store_true', help='Run build + calibrate + evaluate')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--datastore_dir', type=str, default='./datastore')
    
    # Device
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    
    # Set device
    if args.no_cuda or not torch.cuda.is_available():
        args.device = torch.device('cpu')
        print("Using CPU")
    else:
        args.device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    
    if args.run_all:
        args.build_datastore = True
        args.calibrate = True
        args.evaluate = True
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.datastore_dir, exist_ok=True)
    
    # Load model
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)
    model, tokenizer = load_model(args)
    
    # ── STEP 1: Build Datastore ──────────────────────────────
    datastore = KNNDatastore()
    
    if args.build_datastore:
        print("\n" + "=" * 60)
        print("STEP 1: Building Datastore")
        print("=" * 60)
        
        train_dataset = CodeDataset(args.train_data, tokenizer, args.block_size)
        datastore.build(
            model, train_dataset, args.device,
            batch_size=args.batch_size,
            strategy=args.embedding_strategy
        )
        datastore.save(args.datastore_dir)
    else:
        if os.path.exists(os.path.join(args.datastore_dir, 'faiss_index.bin')):
            datastore.load(args.datastore_dir)
        else:
            print("ERROR: No datastore found. Run with --build_datastore first.")
            sys.exit(1)
    
    # ── STEP 2: Calibrate ────────────────────────────────────
    calibrator = TemperatureScaler()
    
    if args.calibrate and args.eval_data:
        print("\n" + "=" * 60)
        print("STEP 2: Calibrating with Temperature Scaling")
        print("=" * 60)
        
        eval_dataset = CodeDataset(args.eval_data, tokenizer, args.block_size)
        eval_outputs = extract_all_outputs(
            model, eval_dataset, args.device,
            batch_size=args.batch_size,
            strategy=args.embedding_strategy
        )
        
        calibrator.fit(eval_outputs['logits'], eval_outputs['labels'])
        calibrator.save(os.path.join(args.output_dir, 'temperature.npy'))
    elif os.path.exists(os.path.join(args.output_dir, 'temperature.npy')):
        calibrator.load(os.path.join(args.output_dir, 'temperature.npy'))
    
    # ── STEP 3: Evaluate on Test Set ─────────────────────────
    if args.evaluate:
        print("\n" + "=" * 60)
        print("STEP 3: Evaluating on Test Set")
        print("=" * 60)
        
        test_dataset = CodeDataset(args.test_data, tokenizer, args.block_size)
        test_outputs = extract_all_outputs(
            model, test_dataset, args.device,
            batch_size=args.batch_size,
            strategy=args.embedding_strategy
        )
        
        # Create predictor
        knn_pred = KNNPredictor(
            datastore=datastore,
            num_classes=args.num_labels,
            k=args.k,
            lambda_val=args.lambda_val,
            knn_temperature=args.knn_temperature,
            voting=args.voting
        )
        
        # Run evaluation
        evaluator = KNNEvaluator(
            num_classes=args.num_labels,
            output_dir=args.output_dir
        )
        
        results = evaluator.run_evaluation(
            model_probs=test_outputs['probs'],
            model_logits=test_outputs['logits'],
            labels=test_outputs['labels'],
            knn_predictor=knn_pred,
            query_embeddings=test_outputs['embeddings'],
            calibrator=calibrator if calibrator.fitted else None
        )
        
        # Save predictions in CodeImprove format
        best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
        print(f"\nBest method: {best_method} (Acc: {results[best_method]['accuracy']:.4f})")
        
        # Save M1 predictions if available, otherwise B4
        if 'M1_knn_calibrated_gated' in results:
            _, m1_preds, _, _ = knn_pred.predict(
                test_outputs['embeddings'],
                calibrator.calibrate(test_outputs['logits']),
                uncertainty_gated=True,
                calibrated_entropy=compute_entropy(calibrator.calibrate(test_outputs['logits']))
            )
            evaluator.save_predictions(
                m1_preds, test_outputs['ids'],
                os.path.join(args.output_dir, 'predictions_knn.txt')
            )
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
