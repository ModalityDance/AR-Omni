from __future__ import annotations

import argparse
import logging
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from datasets import load_dataset
from swanlab import login as swanlab_login
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer, TrainingArguments
from transformers import ChameleonForCausalLM  # Replace if you use a custom model

from perception import PerceptionLoss
from trainer import WeightedTrainer

PAD_TOKEN_ID = 2
IGNORE_TOKEN_ID = -100

# Global max length set by CLI.
MAX_LENGTH: int | None = None

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ParquetDataset(Dataset):
    """
    Memory-mapped Parquet dataset.

    If max_length is provided (>0), samples with len(input_ids) > max_length are filtered out.
    """

    def __init__(self, data_path: str, max_length: int | None = None):
        cache_dir = os.path.join(os.path.dirname(data_path), ".hf_cache")
        logger.info("Loading parquet dataset from %s ...", data_path)

        ds = load_dataset(
            "parquet",
            data_files=data_path,
            split="train",
            cache_dir=cache_dir,
            num_proc=os.cpu_count(),
        )

        if max_length is not None and max_length > 0:
            logger.info("Filtering samples with input_ids length > %s ...", max_length)
            before = len(ds)
            ds = ds.filter(lambda ex: len(ex["input_ids"]) <= max_length, num_proc=os.cpu_count())
            after = len(ds)
            logger.info("Filtered out %s over-length samples; kept %s samples.", before - after, after)

        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def collate_fn(batch):
    """
    Pad variable-length sequences to the same length in a batch.

    Samples are already filtered by max_length (if enabled), so no truncation is performed here.
    """
    input_ids = [ex["input_ids"] for ex in batch]
    labels = [ex["labels"] for ex in batch]
    lengths = [seq.size(0) for seq in input_ids]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)

    max_len = input_ids_padded.size(1)
    attention_mask = (torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training with segment-weighted loss + optional perception loss.")
    parser.add_argument("--data_path", required=True, help="Path to the Parquet data file.")
    parser.add_argument("--model_path", required=True, help="Path to the HF model/weights directory.")
    parser.add_argument("--output_path", required=True, help="Output directory to save the fine-tuned model.")
    parser.add_argument("--deepspeed_config", required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    # SwanLab arguments (DO NOT hardcode keys in open-source code)
    parser.add_argument(
        "--sl_key",
        type=str,
        default="",
        help="SwanLab API key (optional). If empty, uses local cached login or environment configuration.",
    )
    parser.add_argument("--sl_project", required=True, help="SwanLab project name.")
    parser.add_argument("--sl_experiment", required=True, help="SwanLab experiment name.")

    # max_length filter
    parser.add_argument("--max_length", type=int, default=None, help="Maximum input length; longer samples are dropped.")

    # Segment-weighted loss settings (set segment_loss_weight=1.0 to disable)
    parser.add_argument("--segment_start_id", type=int, default=12840, help="Segment start token id.")
    parser.add_argument("--segment_end_id", type=int, default=12841, help="Segment end token id.")
    parser.add_argument("--segment_loss_weight", type=float, default=1.0, help="Loss weight multiplier inside segments.")

    # PerceptionLoss settings + image span settings
    parser.add_argument("--embed_path", type=str, default="perception.ckpt", help="Embedding checkpoint for PerceptionLoss.")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--generator_hidden_size", type=int, default=4096)
    parser.add_argument("--global_weight", type=float, default=1.0, help="Global (perception) loss weight.")
    parser.add_argument("--image_start_id", type=int, default=8197)
    parser.add_argument("--image_end_id", type=int, default=8196)  # kept for API compatibility; not used in logic
    parser.add_argument("--image_seq_length", type=int, default=1024)

    return parser.parse_args()


def main():
    global MAX_LENGTH
    args = parse_args()
    MAX_LENGTH = args.max_length

    # Distributed (optional)
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    # SwanLab login (optional)
    if args.sl_key:
        swanlab_login(api_key=args.sl_key, save=True)

    # Model & tokenizer (tokenizer is kept for downstream compatibility; loading does not change behavior)
    model = ChameleonForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    _ = AutoTokenizer.from_pretrained(args.model_path)

    dataset = ParquetDataset(args.data_path, max_length=MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        bf16=True,
        logging_steps=1,
        save_steps=2000,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed=args.deepspeed_config,
        report_to="none",
        dataloader_drop_last=True,
        dispatch_batches=True,
        ddp_find_unused_parameters=False,
        local_rank=args.local_rank,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
    )

    loss_net = PerceptionLoss(
        embed_path=args.embed_path,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        generator_hidden_size=args.generator_hidden_size,
    )

    swanlab_cb = SwanLabCallback(project=args.sl_project, experiment_name=args.sl_experiment)

    trainer = WeightedTrainer(
        loss_net=loss_net,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[swanlab_cb],
        seg_start_id=args.segment_start_id,
        seg_end_id=args.segment_end_id,
        seg_weight=args.segment_loss_weight,
        global_weight=args.global_weight,
        image_start_id=args.image_start_id,
        image_end_id=args.image_end_id,
        image_seq_length=args.image_seq_length,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(args.output_path)
    logger.info("Model saved to %s", args.output_path)


if __name__ == "__main__":
    main()
