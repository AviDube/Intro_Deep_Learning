import os
os.environ['KAGGLE_USERNAME'] = "avidube" # TODO: Verify in Settings
os.environ['KAGGLE_API_TOKEN'] = "KGAT_e4a59dacaeaefc1cf225368be254aeae"

# Verify
import kaggle
api = kaggle.api  # Already authenticated on import
api.dataset_download_files(
    "cmu11785/s26-11785-hw4-data",
    path=f"{os.environ['LOCAL']}/dataset",
    unzip=True
)

from hw4lib.data import (
    H4Tokenizer,
    ASRDataset
)
from hw4lib.model import (
    DecoderOnlyTransformer,
    EncoderDecoderTransformer
)
from hw4lib.utils import (
    create_scheduler,
    create_optimizer
)
from hw4lib.trainers import (
    ASRTrainer,
    ProgressiveTrainer
)
from torch.utils.data import DataLoader
import yaml
import gc
import torch
from torchinfo import summary
import os
import json
import wandb
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

Tokenizer = H4Tokenizer(
    token_map  = config['tokenization']['token_map'],
    token_type = config['tokenization']['token_type']
)

train_dataset = ASRDataset(
    partition=config['data']['train_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=True,
    global_stats=None  # Will compute stats from training data
)

# TODO: Get the computed global stats from training set
global_stats = None
if config['data']['norm'] == 'global_mvn':
    global_stats = (train_dataset.global_mean, train_dataset.global_std)
    print(f"Global stats computed from training set.")

val_dataset = ASRDataset(
    partition=config['data']['val_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=False,
    global_stats=global_stats
)

test_dataset = ASRDataset(
    partition=config['data']['test_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=False,
    global_stats=global_stats
)

gc.collect()

train_loader    = DataLoader(
    dataset     = train_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = True,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = train_dataset.collate_fn
)

val_loader      = DataLoader(
    dataset     = val_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = False,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = val_dataset.collate_fn
)

test_loader     = DataLoader(
    dataset     = test_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = False,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn
)

gc.collect()

max_feat_len       = max(train_dataset.feat_max_len, val_dataset.feat_max_len, test_dataset.feat_max_len)
max_transcript_len = max(train_dataset.text_max_len, val_dataset.text_max_len, test_dataset.text_max_len)
max_len            = max(max_feat_len, max_transcript_len)

print("="*50)
print(f"{'Max Feature Length':<30} : {max_feat_len}")
print(f"{'Max Transcript Length':<30} : {max_transcript_len}")
print(f"{'Overall Max Length':<30} : {max_len}")
print("="*50)

wandb.login(key="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")

model_config = config['model'].copy()
model_config.update({
    'max_len': max_len,
    'num_classes': Tokenizer.vocab_size
})

model = EncoderDecoderTransformer(**model_config)

# Get some inputs from the train dataloader
for batch in train_loader:
    padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths = batch
    break

total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert total_param < 30_000_000, f"Total trainable parameters ({total_param}) exceeds 30 million."

model_stats = summary(model, input_data=[padded_feats, padded_shifted, feat_lengths, transcript_lengths])
print(model_stats)

trainer = ASRTrainer(
    model=model,
    tokenizer=Tokenizer,
    config=config,
    run_name=config['training']['run_name'],
    config_file="config.yaml",
    device=device
)

print("Creating optimizer...")
trainer.optimizer = create_optimizer(
    model=model,
    opt_config=config['optimizer']
)

trainer.scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)

# Resume if needed
if config['training']['resume'] and config['training']['wandb_run_id'] != 'none':
    trainer.load_checkpoint('checkpoint-last-epoch-model.pth')
    print(f"Resumed from epoch {trainer.current_epoch}")

remain_epochs = config['training']['epochs'] - trainer.current_epoch
print(f"Remaining epochs to train: {remain_epochs}")
# trainer.train(train_loader, val_loader, epochs=remain_epochs)

# Define the recognition config: Greedy search
recognition_config = {
    'num_batches': None,
    'temperature': 1.0,
    'repeat_penalty': 1.0,
    'lm_weight': 0.0,
    'lm_model': None,
    'beam_width': 12,
}

# Recognize with the shallow fusion config
config_name = "test"
print(f"Evaluating with {config_name} config")
results = trainer.recognize(test_loader, recognition_config, config_name=config_name, max_length=max_transcript_len)


# Calculate metrics on full batch
generated = [r['generated'] for r in results]
results_df = pd.DataFrame(
    {
        'id': range(len(generated)),
        'transcription': generated
    }
)

# Cleanup (Will end wandb run)
trainer.cleanup()

results_df.to_csv("results.csv", index=False)
api.competition_submit(file_name="results.csv", message="submission for HW4P2", competition="11785-hw-4-p-2-asr-with-transformer-spring-2026-slack")