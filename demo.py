# Demo.py
import argparse
import sys
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
from tqdm import tqdm
import math
import re

# ================================== Custom DECODER-ONLY MODEL for M8 ====================================
class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, max_seq_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) + self.pos_embedding(positions)
        tgt_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        memory = torch.zeros((seq_length, batch_size, x.size(-1)), device=x.device)
        x = self.transformer_decoder(x.permute(1, 0, 2), memory, tgt_mask=tgt_mask)
        return self.fc_out(x.permute(1, 0, 2))

# ============================== Custom ENCODE-DECODER MODEL from scratch ================================

# Sinusoidal Positional Encoding
def get_sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]

# ==========================
# Model Components
# ==========================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        B, T, D = query.size()
        H = self.num_heads

        def reshape(x):
            return x.view(B, -1, H, self.head_dim).transpose(1, 2)

        Q = reshape(self.q_proj(query))
        K = reshape(self.k_proj(key))
        V = reshape(self.v_proj(value))

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.ff(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt

class CustomEncoderDecoderModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=2048, num_layers=4, dropout=0.1, vocab_size=32100, max_len=512):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.positional_encoding = get_sinusoidal_encoding(max_len, embed_dim)

    def forward(self, encoder_embeddings, decoder_embeddings, src_mask=None, tgt_mask=None):
        B, S, _ = encoder_embeddings.size()
        B2, T, _ = decoder_embeddings.size()

        # Add sinusoidal positional embeddings
        encoder_embeddings = encoder_embeddings + self.positional_encoding[:, :S, :].to(encoder_embeddings.device)
        decoder_embeddings = decoder_embeddings + self.positional_encoding[:, :T, :].to(decoder_embeddings.device)

        memory = encoder_embeddings
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        if tgt_mask is None:
            tgt_mask = torch.tril(torch.ones(T, T)).to(decoder_embeddings.device)  # [T, T]
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        output = decoder_embeddings
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, src_mask)

        logits = self.lm_head(output)
        return logits

# -------------- Inference for Custom Encoder-Decoder Model ---------------
class CustomEncoderDecoderSummaryGenerator:
    def __init__(self, model, tokenizer, embedding_layer, device, max_input_len=256, max_target_len=80):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer.to(device)
        self.device = device
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def top_k_sampling(self, logits, k=50, temperature=1.0):
        logits = logits / temperature
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(top_k_values, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return top_k_indices.gather(-1, sampled_idx)

    def generate_summary(self, code_snippet):
        code_snippet = re.sub(r'\s+', ' ', code_snippet).strip()
        source = self.tokenizer(code_snippet,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_input_len,
                                return_tensors="pt").to(self.device)

        with torch.no_grad():
            encoder_embeddings = self.embedding_layer(source["input_ids"])
            memory = encoder_embeddings
            for layer in self.model.encoder_layers:
                memory = layer(memory, source["attention_mask"])

        generated_ids = torch.full((1, 1), self.tokenizer.pad_token_id, dtype=torch.long).to(self.device)

        for _ in range(self.max_target_len):
            with torch.no_grad():
                decoder_embeddings = self.embedding_layer(generated_ids)
                logits = self.model(memory, decoder_embeddings)
                next_token_logits = logits[:, -1, :]
                # To top-k sampling:
                next_token_id = self.top_k_sampling(next_token_logits, k=50, temperature=0.7)
                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

                if next_token_id.item() == self.tokenizer.pad_token_id:
                    break
                if (generated_ids[0, -10:] == next_token_id).all():
                    print("⚠️ Stopping early due to repetition")
                    break

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

# -------------- Inference for Finetuned Encoder-Decoder Model ---------------
class CodeT5SummaryGenerator:
    def __init__(self, model_path: str, decoding_config: Dict, device: Optional[str] = None):
        self.model_path = model_path
        self.decoding_config = decoding_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def generate_single(self, code_snippet: str):
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, **self.decoding_config)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

def get_custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        'pad_token': '<pad>',
        'sep_token': '<sep>',
        'bos_token': '<s>',
        'eos_token': '</s>'
    })
    return tokenizer


def load_custom_model(model_path="M8_final_model.pth", d_model=768, n_heads=8, n_layers=6):
    tokenizer = get_custom_tokenizer()
    model = DecoderOnlyModel(len(tokenizer), d_model, n_heads, n_layers)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, tokenizer


def generate_docstring_custom_model(model, tokenizer, code_snippet, max_length=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_text = f"<s> {code_snippet} <sep>"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text.split("<sep>")[-1].strip()


def generate_docstring_few_shot(test_code, model_choice):
    if model_choice == "1":
        model_path = "./gpt2-docstring-model"
        few_shot_prompt = """
<s> public int add(int a, int b) { return a + b; } </s> <sep> Adds two integers and returns the sum.

<s> public int multiply(int a, int b) { return a * b; } </s> <sep> Multiplies two integers and returns the product.

<s> public boolean isEven(int num) { return num % 2 == 0; } </s> <sep> Checks if a number is even.

<s> public String greet(String name) { return "Hello " + name; } </s> <sep> Greets the user by name.
"""
        prompt = few_shot_prompt.strip() + f"\n<s> {test_code} </s> <sep>"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_len = input_ids.shape[1]

        output_ids = model.generate(
            input_ids,
            max_length=input_len + 50,
            num_beams=9,
            no_repeat_ngram_size=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_ids = output_ids[0][input_len:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    elif model_choice == "2":
        model, tokenizer = load_custom_model("M8_final_model.pth")
        return generate_docstring_custom_model(model, tokenizer, test_code)

    else:
        raise ValueError("Invalid model choice")


def handle_encoder_decoder_summary(code_snippet: str, choice: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if choice == "3":
        print("🔧 Using Fine-tuned CodeT5 from Hugging Face")
        model_path = "pritammane105/CodeT5-Java-Summarisation"
        decoding_config = {
            "max_new_tokens": 64,
            "do_sample": True,
            "top_k": 50,
            "temperature": 0.7,
            "repetition_penalty": 1.2
        }
        generator = CodeT5SummaryGenerator(model_path=model_path, decoding_config=decoding_config, device=device)
        return generator.generate_single(code_snippet)
    
    elif choice == "4":
        print("🔧 Using Custom Encoder-Decoder Transformer Model on Hugging Face")

        model_path = "pritammane105/CodeT5-Java-Summarisation"
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        embedding_layer = AutoModel.from_pretrained("Salesforce/codet5-base").get_input_embeddings()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download model checkpoint from Hugging Face Hub
        checkpoint_path = hf_hub_download(
            repo_id="pritammane105/Custom-Java-Summarisation",
            filename="my_model.pt"
        )
        model = CustomEncoderDecoderModel()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        embedding_layer.to(device)
        model.eval()

        generator = CustomEncoderDecoderSummaryGenerator(model, tokenizer, embedding_layer, device)
        return generator.generate_summary(code_snippet)

    else:
        print("Invalid choice")
        sys.exit(1)

def main():
    print("=== Docstring Generator ===")
    print("Choose a model:")
    print("1. Model 1 (Fine-tuned GPT2: Decoder-Only)")
    print("2. Model 2 (Custom Model: Decoder-Only)")
    print("1. Model 3 (Fine-tuned CodeT5: Encoder-Decoder)")
    print("2. Model 4 (Custom Model: Encoder-Decoder)")

    choice = input("Enter your choice (1 or 2): ").strip()

    if not choice:
        print("Error: You must provide a choice.")
        sys.exit(1)

    if choice not in ["1", "2", "3", "4"]:
        print("Error: Invalid choice.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate docstring for Java method.")
    parser.add_argument("--code", type=str, required=True, help="Java method code as a string.")
    args = parser.parse_args()

    if choice in ["1", "2"]:
        docstring = generate_docstring_few_shot(args.code, choice)
    else:
        docstring = handle_encoder_decoder_summary(args.code, choice)
    
    print("\n Generated Docstring:\n", docstring)


if __name__ == "__main__":
    main()
