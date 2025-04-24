
# 🧠 Code Summarizer — Java Method Docstring Generator

This project generates concise docstrings for Java methods using transformer-based models. It supports both encoder-decoder (CodeT5) and decoder-only (GPT2) architectures — including fine-tuned and custom-trained models.

---

## 📂 Project Structure

```
.
├── demo.py                          # 🔧 CLI demo script
├── decoder_only_models/
│   └── gpt2_training.ipynb          # Notebook for GPT2 training
├── encoder_decoder_models/
│   ├── codet5_finetune_train.ipynb  # Notebook to fine-tune CodeT5
│   ├── codet5_finetune_inference.ipynb  # Run inference & generate metrics
│   └── custom_encoder_decoder.ipynb  # Run inference & generate metrics
└── README.md                        # This file
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

To evaluate different decoding strategies and analyze model performance:

### 👉 Open this notebook in Google Colab:

```
encoder_decoder_models/codet5_finetune_inference.ipynb
```

It allows you to:
- Run inference with multiple decoding configs (beam search, sampling)
- Save results as `.csv`
- Compute metrics: ROUGE, BLEU, BERTScore, repetition
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

- **You**: Fine-tuned CodeT5, custom encoder-decoder
- **Teammate**: GPT2 training & custom decoder-only architecture

---

## 🗂 External Resources

- 🤗 Hugging Face Model: [`pritammane105/CodeT5-Java-Summarisation`](https://huggingface.co/pritammane105/CodeT5-Java-Summarisation)

---

## 📌 Notes

- Models support GPU if available (automatically detected via PyTorch)
- Results are saved in Google Drive (Colab) or locally as CSV
- Modular design for adding new decoding strategies

---

## 📬 Contact

Feel free to raise an issue or reach out via GitHub if you have questions or feedback.
