import pytest
import torch
import re
from transformers import GPT2Tokenizer

# Load tokenizer (update the path if needed)
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-docstring-model")
tokenizer.pad_token = tokenizer.eos_token  # ensure pad token is set

# 1. Tokenization works correctly
def test_tokenizer_output_keys():
    sample = "public int add(int a, int b) { return a + b; }"
    result = tokenizer(sample, return_tensors="pt")
    assert "input_ids" in result and "attention_mask" in result

# 2. Input length constraint
def test_input_truncation():
    sample = "int x = 0; " * 300  # Create a long code string
    tokens = tokenizer(sample, max_length=512, truncation=True)
    assert len(tokens["input_ids"]) <= 512

# 3. Label length constraint
def test_label_truncation():
    docstring = "This is a docstring. " * 30
    tokens = tokenizer(docstring, max_length=128, truncation=True)
    assert len(tokens["input_ids"]) <= 128

# 4. Padding mask correctness
def test_label_padding_mask():
    tokens = tokenizer("example text", padding="max_length", max_length=10)
    input_ids = tokens["input_ids"]
    labels = [token if token != tokenizer.pad_token_id else -100 for token in input_ids]
    for i, token in enumerate(input_ids):
        if token == tokenizer.pad_token_id:
            assert labels[i] == -100

# 5. Whitespace normalization in prediction
def test_whitespace_normalization():
    example = "This    is\ta test.\n"
    normalized = re.sub(r"\\s+", " ", example).strip()
    assert normalized == "This is a test."

if __name__ == "__main__":
    pytest.main(["-v"])
