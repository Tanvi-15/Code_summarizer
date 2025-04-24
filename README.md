
# 🧠 Code Summarizer — Java Method Docstring Generator

This project generates concise docstrings for Java methods using transformer-based models. It supports both encoder-decoder (CodeT5) and decoder-only (GPT2) architectures — including fine-tuned and custom-trained models.

---

## 📂 Project Structure

```
.
├── demo.py                              # 🔧 CLI demo script
├── decoder_only_models/
│   └── gpt2_training.ipynb              # Notebook for GPT2 training
├── encoder_decoder_models/
│   ├── codet5_finetune_train.ipynb      # Notebook to fine-tune CodeT5
│   ├── codet5_finetune_inference.ipynb  # Run inference & generate metrics
│   └── custom_encoder_decoder.ipynb     # Run inference & generate metrics
├── resources/                           # All the csv files with generated summaries on validation & test sets
│   ├── codet5_val_baseline_beam.csv
│   ├── codet5_val_baseline_beam_repetition.csv
│   ├── codet5_val_topk_sampling.csv
│   ├── custom_test_topk_sampling.csv
│   └── custom_val_topk_sampling.csv
└── README.md
```

---

## 🚀 How to Run the Demo

### ✅ Step 1: Install Requirements

```bash
pip install transformers datasets evaluate torch pandas
```

Or, if provided:

```bash
pip install -r requirements.txt
```

---

### ✅ Step 2: Run the CLI Script

```bash
python demo.py --code "public int add(int a, int b) { return a + b; }"
```

You will be prompted to choose one of the models.

---

## 🤖 Model Choices

| Choice | Model Description                              |
|--------|------------------------------------------------|
|   1    | Fine-tuned GPT2 (decoder-only)                 |
|   2    | Custom decoder-only model                      |
|   3    | Fine-tuned CodeT5 (encoder-decoder) ✅         |
|   4    | Custom encoder-decoder model                   |

---

## 📊 Inference & Evaluation (Colab Recommended)

To evaluate different decoding strategies and analyze Finetuned Model performance:

### 👉 Open this notebook in Google Colab:

```
encoder_decoder_models/codet5_finetune_inference.ipynb
```

It allows you to:
- Run batch inference with 3 decoding configs (baseline_beam, beam_repetition_penalty, topk_sampling) on Validation Set & with topk_sampling decoding on Test Set
- Save the result summaries as `.csv`. Make sure to pass a custom path (if needed) to `.generate_summaries()` funtion's third parameter to save the files.
- Compute metrics: ROUGE, BLEU, BERTScore, repetition on the csv files generated previously.
- Display a comparison table

---

To evaluate summaries and analyze Custom Model performance:

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

---

## 📈 Metrics Reported

- **ROUGE-1 / ROUGE-2 / ROUGE-L**
- **BLEU Score**
- **BERTScore**
- **Exact Match Accuracy**
- **Average Token Repetition**

---

## 🤝 Credits

- **Pritam Anand Mane**: Fine-tuned CodeT5, custom encoder-decoder
- **Tanvi Deshpande**: GPT2 training & custom decoder-only architecture

---

## 🗂 External Resources

- Hugging Face Model: [`pritammane105/CodeT5-Java-Summarisation`](https://huggingface.co/pritammane105/CodeT5-Java-Summarisation)
- Hugging Face Model: [`pritammane105/Custom-Java-Summarisation`](https://huggingface.co/pritammane105/Custom-Java-Summarisation)

---

## 📌 Notes

- Models support GPU if available (automatically detected via PyTorch)
- Results are saved in Google Drive (Colab) or locally as CSV
- Modular design for adding new decoding strategies

---

## 📬 Contact

Feel free to raise an issue or reach out via GitHub if you have questions or feedback.
