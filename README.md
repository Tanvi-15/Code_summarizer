
# 🧠 Code Summarizer — Java Method Docstring Generator

This project generates concise docstrings for Java methods using transformer-based models. It supports both encoder-decoder (CodeT5) and decoder-only (GPT2) architectures — including fine-tuned and custom-trained models.

---

## 📂 Project Structure

```
.
├── demo.py                              #  CLI demo script
├── decoder_only_models
│   └── Custom_model.ipynb               # Notebook for custom decoder-only model
│   └── GPT2_Finetune.ipynb              # Notebook for GPT2 finetune training
├── Encoder_Decoder_Models
│   ├── codet5_finetune_training.ipynb   # Notebook for fine-tuning CodeT5
│   └── custom_encoder_decoder.ipynb     # Notebook for custom encoder-decoder training and evaluation
├── Model_inference_and_evaluation
│   └── finetune_models_inference.ipynb  # Run analysis on fine-tuned models
├── resources/                           # All the csv files with generated summaries on validation & test sets
│   └── decoder-only-summaries
│       ├── GPT2_test_sampling_output.csv
│       ├── GPT2_val_beam_repetition.csv
│       ├── GPT2_val_predictions.csv\
│       └── GPT2_val_sampling_topk.csv
│   └── encoder-decoder-summaries
│       ├── codet5_val_baseline_beam.csv
│       ├── codet5_val_beam_repetition.csv
│       ├── codet5_val_topk_sampling.csv
│       ├── custom_test_topk_sampling.csv
│       └── custom_val_topk_sampling.csv
│
├── requirements.txt  
└── README.md
```

---

## 🚀 How to Run the Demo

### ✅ Step 1: Install Requirements (Setup Env)


```bash
pip install -r requirements.txt
```

---

### ✅ Step 2: Run the CLI Script

```bash
python demo.py --code "public int add(int a, int b) { return a + b; }"
```
OR
```bash
python demo.py --code "<Java code snippe in one line>"
```

You will be prompted to choose one of the models.

---

## 🤖 Model Choices

| Choice | Model Description                              |
|--------|------------------------------------------------|
|   1    | Fine-tuned GPT2 (decoder-only)                 |
|   2    | Custom decoder-only model                      |
|   3    | Fine-tuned CodeT5 (encoder-decoder)          |
|   4    | Custom encoder-decoder model                   |

---

## 📊 Inference & Evaluation (Colab Recommended)

To evaluate different decoding strategies and analyze *Finetuned* *Model* performance:

### 👉 Open this notebook in Google Colab:

```
Model_Inference_and_Evaluation/finetune_models_inference.ipynb
```

It allows you to:
- Run batch inference with 3 decoding configs (baseline_beam, beam_repetition_penalty, topk_sampling) on Validation Set & with topk_sampling decoding on Test Set
- Save the result summaries as `.csv`. Make sure to pass a custom path (if needed) to `.generate_summaries()` funtion's third parameter to save the files.
- Compute metrics: ROUGE, BLEU, BERTScore, repetition on the csv files generated previously.
- Display a comparison table

---


To evaluate summaries and analyze *Custom* *Model* *performance*:
For Custom Encoder-decoder model
### 👉 Open this notebook in Google Colab:

```
encoder_decoder_models/custom_encoder_decoder.ipynb
```

Run all the cells in order except the one titled `MODEL TRAINING`

It allows you to:
- Run batch inference with topk_sampling on Validation & Test Sets.
- Save the result summaries as `.csv`. Make sure to pass a custom path (if needed) to `.generate_summaries()` funtion's third parameter to save the files.
- Compute metrics: ROUGE, BLEU, BERTScore, repetition  on the csv files generated previously.
- Display a comparison table


For Custom Decoder Only Model

```
decoder_only_model/Custom_model.ipynb
```
TBC

---

## 📈 Metrics Reported

- **ROUGE-1 / ROUGE-2 / ROUGE-L**
- **BLEU Score**
- **BERTScore**
- **Average Token Repetition**

---

## 🤝 Credits

- **Pritam Anand Mane**: Fine-tuned CodeT5, custom encoder-decoder
- **Tanvi Deshpande**: GPT2 training & custom decoder-only architecture

---

## 🗂 External Resources

- Hugging Face Model: [`pritammane105/CodeT5-Java-Summarisation`](https://huggingface.co/pritammane105/CodeT5-Java-Summarisation)
- Hugging Face Model: [`pritammane105/Custom-Java-Summarisation`](https://huggingface.co/pritammane105/Custom-Java-Summarisation)
- Hugging Face Model: [`pritammane105/GPT2-Code-Summarisation`](https://huggingface.co/pritammane105/GPT2-Code-Summarisation)
---

## 📌 Notes

- Models support GPU if available (automatically detected via PyTorch)
- Results are saved in Google Drive (Colab) or locally as CSV
- Modular design for adding new decoding strategies

---

