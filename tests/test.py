import pytest
import torch
import re
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-docstring-model")
tokenizer.pad_token = tokenizer.eos_token  # set pad token for safe masking

# 1. Tokenization output contains expected keys
def test_tokenizer_output_keys():
    sample = "public int add(int a, int b) { return a + b; }"
    result = tokenizer(sample, return_tensors="pt")
    assert "input_ids" in result and "attention_mask" in result

# 2. Long input gets truncated
def test_input_truncation():
    sample = "int x = 0; " * 300  # long string
    tokens = tokenizer(sample, max_length=512, truncation=True)
    assert len(tokens["input_ids"]) <= 512

# 3. Truncated docstring label stays within max length
def test_label_truncation():
    docstring = "This is a docstring. " * 30
    tokens = tokenizer(docstring, max_length=128, truncation=True)
    assert len(tokens["input_ids"]) <= 128

# 4. Padding token should be masked as -100 in labels
def test_label_padding_mask():
    tokens = tokenizer("example text", padding="max_length", max_length=10)
    input_ids = tokens["input_ids"]
    labels = [token if token != tokenizer.pad_token_id else -100 for token in input_ids]
    for i, token in enumerate(input_ids):
        if token == tokenizer.pad_token_id:
            assert labels[i] == -100

# 5. Whitespace normalization for prediction formatting
def test_whitespace_normalization():
    example = "This    is\ta test.\n"
    normalized = re.sub(r"\s+", " ", example).strip()
    assert normalized == "This is a test."
