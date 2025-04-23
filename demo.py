# Demo.py

import argparse
import sys
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer

# === Custom DecoderOnlyModel for M8 ===
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


def main():
    print("=== Docstring Generator ===")
    print("Choose a model:")
    print("1. Model 1 (HuggingFace fine-tuned GPT-2)")
    print("2. Model 2 (Custom M8 Transformer)")

    choice = input("Enter your choice (1 or 2): ").strip()

    if not choice:
        print("Error: You must provide a choice (1 or 2).")
        sys.exit(1)

    if choice not in ["1", "2"]:
        print("Error: Invalid choice. Please enter 1 or 2.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate docstring for Java method.")
    parser.add_argument("--code", type=str, required=True, help="Java method code as a string.")
    args = parser.parse_args()

    docstring = generate_docstring_few_shot(args.code, choice)
    print("\n✅ Generated docstring:\n", docstring)


if __name__ == "__main__":
    main()
