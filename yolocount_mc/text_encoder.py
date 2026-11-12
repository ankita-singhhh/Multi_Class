
"""
CLIP-based text encoder for YOLO-Count multi-class counting.

Input:
    List[List[str]]

Output:
    Tensor [B, K, d]

Uses pretrained CLIP text representations.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        d: int = 512,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
        normalize: bool = False,
    ):
        super().__init__()

        if d <= 0:
            raise ValueError("Embedding dimension d must be positive.")

        try:
            from transformers import CLIPModel, CLIPTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install transformers: pip install transformers"
            ) from exc

        self.output_dim = d
        self.normalize = normalize
        self.freeze = freeze
        self.model_name = model_name

        # Legacy compatibility only; CLIP does not use this vocabulary.
        self.vocab = (
            {word: index for index, word in enumerate(vocab)}
            if vocab is not None
            else {}
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name)

        clip_dim = self.clip.config.projection_dim

        if clip_dim == d:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Linear(clip_dim, d)

        if self.freeze:
            for parameter in self.clip.parameters():
                parameter.requires_grad = False

            self.clip.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        # Frozen CLIP should not update its training behavior.
        if self.freeze:
            self.clip.eval()

        return self

    def forward(self, queries: List[List[str]]) -> torch.Tensor:
        if not isinstance(queries, list) or not queries:
            raise ValueError("queries must be a non-empty list.")

        if not all(isinstance(row, list) for row in queries):
            raise TypeError("Each batch item must be a list of class names.")

        batch_size = len(queries)
        num_queries = len(queries[0])

        if num_queries == 0:
            raise ValueError("Each image must contain at least one query.")

        if any(len(row) != num_queries for row in queries):
            raise ValueError(
                "All batch items must have the same number of queries. "
                "Pad queries in the dataset collate function."
            )

        flattened_queries = []

        for row in queries:
            for query in row:
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(
                        "Every class query must be a non-empty string."
                    )
                flattened_queries.append(query.strip())

        device = next(self.clip.parameters()).device

        tokens = self.tokenizer(
            flattened_queries,
            padding=True,
            truncation=True,
            max_length=self.clip.config.text_config.max_position_embeddings,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Avoid building an autograd graph for frozen CLIP.
        if self.freeze:
            with torch.no_grad():
                text_output = self.clip.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                pooled_output = text_output.pooler_output
                embeddings = self.clip.text_projection(pooled_output)
        else:
            text_output = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pooled_output = text_output.pooler_output
            embeddings = self.clip.text_projection(pooled_output)

        embeddings = self.output_projection(embeddings)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        embeddings = embeddings.reshape(
            batch_size,
            num_queries,
            self.output_dim,
        )

        return embeddings
