from __future__ import annotations

import argparse
import logging
import os

import jsonlines
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import ChameleonForCausalLM, TrainingArguments

from perception import PerceptionLoss
from pretrain_trainer import PretrainWeightedAndPerceptionTrainer
from swanlab.integration.transformers import SwanLabCallback

VOCAB_SIZE = 8192
HIDDEN_SIZE = 256
GENERATOR_HIDDEN_SIZE = 4096

PAD_ID = 2
IGNORE_TOKEN_ID = -100

logger = logging.getLogger(__name__)


class MultiShardIterableDataset(IterableDataset):
    def __init__(self, dataset_dir: str, skip_shards: int = 0, skip_samples: int = 0) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.skip_shards = int(skip_shards)
        self.skip_samples = int(skip_samples)

        self.shard_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jsonl")])
        assert self.shard_files, "No dataset shards found in directory!"

        self.estimated_samples_per_shard = self._count_samples_in_one_shard()
        self.total_samples = self.estimated_samples_per_shard * len(self.shard_files)

    def _count_samples_in_one_shard(self) -> int:
        first_shard = self.shard_files[0]
        first_shard_path = os.path.join(self.dataset_dir, first_shard)
        count = 0
        with jsonlines.open(first_shard_path, "r") as reader:
            for obj in reader:
                if "discrete_code" in obj and 0 < len(obj["discrete_code"]) < 1300:
                    count += 1
        return count

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self):
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        shard_files_to_process = self.shard_files[self.skip_shards :]
        samples_to_skip_in_current_shard = self.skip_samples

        for shard_file in shard_files_to_process:
            shard_path = os.path.join(self.dataset_dir, shard_file)
            with jsonlines.open(shard_path, "r") as reader:
                offset = 0
                for obj in reader:
                    if samples_to_skip_in_current_shard > 0:
                        samples_to_skip_in_current_shard -= 1
                        continue

                    if "discrete_code" in obj:
                        seq = obj["discrete_code"]
                        if 0 < len(seq) < 1300:
                            task = obj.get("task", "")
                            resp_len = len(obj.get("response", []) or [])

                            if offset % 20 == 0:
                                logger.info(
                                    "rank=%s, shard=%s, offset=%s, len=%s, task=%s",
                                    rank,
                                    shard_file,
                                    offset,
                                    len(seq),
                                    task,
                                )
                            offset += 1

                            yield {
                                "input_ids": torch.tensor(seq, dtype=torch.long),
                                "task": task,
                                "response_len": resp_len,
                                "seq_len": len(seq),
                                "id": obj.get("id", None),
                            }

            samples_to_skip_in_current_shard = 0


TASK_VOCAB = [
    "image_caption",
    "speech_to_text",
    "text_to_image",
    "text_to_speech",
    "image_to_text",
    "speech_to_image",
    "text_to_text",
]
TASK2ID = {n: i for i, n in enumerate(TASK_VOCAB)}
PACK_BASE = 2048


def _pack_meta(task_id: torch.Tensor, resp_len: torch.Tensor) -> torch.Tensor:
    return (task_id + 1) * PACK_BASE + resp_len


def collate_fn_pretrain(batch):
    input_ids_list = [b["input_ids"] for b in batch]
    seq_lens = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    resp_lens = torch.tensor([b["response_len"] for b in batch], dtype=torch.long)
    task_ids = torch.tensor([TASK2ID[b["task"]] for b in batch], dtype=torch.long)

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=PAD_ID)
    B, T = input_ids.size()

    arange = torch.arange(T).unsqueeze(0).expand(B, T)
    attention_mask = (arange < seq_lens.unsqueeze(1)).long()

    labels = input_ids.clone()
    labels[arange >= seq_lens.unsqueeze(1)] = IGNORE_TOKEN_ID

    labels[:, 0] = _pack_meta(task_ids, resp_lens)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tasks": [b["task"] for b in batch],
        "seq_lens": seq_lens,
        "resp_lens": resp_lens,
        "task_ids": task_ids,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain with response-weighted loss + optional perception loss (text_to_image)."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--skip_shards", type=int, default=0)
    parser.add_argument("--skip_samples", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--swan_project", type=str, default="")
    parser.add_argument("--swan_experiment", type=str, default="")

    parser.add_argument("--response_weighted_tasks", type=str, default="image_caption,speech_to_text")
    parser.add_argument("--response_seg_weight", type=float, default=2.0)

    parser.add_argument("--perception_weight", type=float, default=1.0)
    parser.add_argument("--image_start_id", type=int, default=8197)
    parser.add_argument("--image_end_id", type=int, default=8196)
    parser.add_argument("--image_seq_length", type=int, default=1024)

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    model = ChameleonForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    train_dataset = MultiShardIterableDataset(
        dataset_dir=args.dataset_dir,
        skip_shards=args.skip_shards,
        skip_samples=args.skip_samples,
    )

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        max_steps=150000,
        per_device_train_batch_size=2,
        fp16=False,
        bf16=True,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=1000,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed=args.deepspeed_config,
        dataloader_drop_last=True,
        dispatch_batches=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        local_rank=args.local_rank,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        ignore_data_skip=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    loss_net = None
    if args.perception_weight != 0:
        embed_path = "perception.ckpt"
        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"PerceptionLoss embed weights not found: {embed_path}")
        loss_net = PerceptionLoss(
            embed_path=embed_path,
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            generator_hidden_size=GENERATOR_HIDDEN_SIZE,
        )

    swan_cb = SwanLabCallback(project=args.swan_project, experiment_name=args.swan_experiment)

    trainer = PretrainWeightedAndPerceptionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn_pretrain,
        callbacks=[swan_cb],
        response_weighted_tasks=set([s.strip() for s in args.response_weighted_tasks.split(",") if s.strip()]),
        response_seg_weight=args.response_seg_weight,
        loss_net=loss_net if args.perception_weight != 0 else None,
        perception_weight=args.perception_weight,
        image_start_id=args.image_start_id,
        image_end_id=args.image_end_id,
        image_seq_length=args.image_seq_length,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    if rank == 0:
        trainer.save_model(args.output_path)
        logger.info("Model saved to %s", args.output_path)


if __name__ == "__main__":
    main()
