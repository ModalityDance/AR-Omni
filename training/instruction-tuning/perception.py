from __future__ import annotations

import torch
import torch.nn as nn


class PerceptionLoss(nn.Module):
    """
    Compute perception loss on a fixed-length token sequence (length=1024).

    Notes:
        - The embedding table is loaded from `embed_path` and frozen.
        - `input_ids - 4` is preserved to match the original token mapping.
    """

    def __init__(
        self,
        embed_path: str = "perception.ckpt",
        vocab_size: int = 8192,
        hidden_size: int = 256,
        generator_hidden_size: int = 4096,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)

        embedding = torch.load(embed_path, map_location="cpu")
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True).to(dtype=torch.bfloat16)
        self.states_to_hidden = nn.Linear(generator_hidden_size, hidden_size, dtype=torch.bfloat16)
        self.criterion = nn.MSELoss()

    def forward(self, input_ids: torch.Tensor, generated_hidden_states: torch.Tensor) -> torch.Tensor:
        assert input_ids.shape[1] == 1024, f"Expected input_ids length=1024, got {input_ids.shape[1]}"
        assert (
            generated_hidden_states.shape[1] == 1024
        ), f"Expected generated_hidden_states length=1024, got {generated_hidden_states.shape[1]}"

        labels = self.embedding(input_ids - 4)
        features = self.states_to_hidden(generated_hidden_states)
        return self.criterion(features, labels)
