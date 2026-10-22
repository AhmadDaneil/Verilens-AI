# tokenizer.py
# WordPiece tokenizer — mirrors tokenizer_service.dart exactly.
# Uses HuggingFace tokenizer for correctness and speed.

from transformers import DistilBertTokenizerFast
import torch


class Tokenizer:
    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self._tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(self, text: str) -> dict:
        """
        Returns input_ids and attention_mask as torch tensors of shape [1, 128].
        Matches the tokenizer_service.dart output exactly.
        """
        enc = self._tok(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"],       # [1, 128] int64
            "attention_mask": enc["attention_mask"],   # [1, 128] int64
        }