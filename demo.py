# Demo.py

import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys

def generate_docstring_few_shot(test_code, model_path):
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
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return result

def main():
    print("=== Docstring Generator ===")
    print("Choose a model:")
    print("1. Model 1")
    print("2. Model 2")

    choice = input("Enter your choice (1 or 2): ").strip()

    if not choice:
        print("Error: You must provide a choice (1 or 2).")
        sys.exit(1)

    if choice == "1":
        model_path = "./gpt2-docstring-model"
    # elif choice == "2":
    #     model_path = "./gpt2-docstring-model-2"
    else:
        print("Error: Invalid choice. Please enter 1 or 2.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate docstring for Java method.")
    parser.add_argument("--code", type=str, required=True, help="Java method code as a string.")
    args = parser.parse_args()

    docstring = generate_docstring_few_shot(args.code, model_path)
    print("\n✅ Generated docstring:\n", docstring)

if __name__ == "__main__":
    main()
