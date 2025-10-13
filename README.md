# Fine-Tuning an English-to-Turkish Translation Model 🇹🇷

This repository contains a Jupyter Notebook demonstrating the process of fine-tuning the `Helsinki-NLP/opus-mt-tc-big-en-tr` model for English-to-Turkish machine translation. The fine-tuning was performed on the `kde4` dataset, resulting in a significant improvement in translation quality, as measured by the BLEU score.

The final, fine-tuned model is available on the Hugging Face Hub:
**[gokhanErgul/kde4-en-to-tr_with-Helsinki](https://huggingface.co/gokhanErgul/kde4-en-to-tr_with-Helsinki)**

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/gokhanErgul/kde4-en-to-tr_with-Helsinki)

---
## 📊 Performance Improvement

Fine-tuning led to a substantial increase in the model's performance on the test set. The **BLEU score more than doubled**, indicating a much higher quality of translation.

| Stage                 | BLEU Score |
| :-------------------- | :--------- |
| Before Fine-Tuning    | 13.69      |
| **After Fine-Tuning** | **28.94** |

---
## 🚀 Project Workflow

The project followed a standard procedure for fine-tuning a sequence-to-sequence model using the Hugging Face ecosystem.

1.  **Dataset Preparation**: The `kde4` dataset with the English (`en`) and Turkish (`tr`) language pair was loaded. It was then split into a **90% training set** and a **10% test set**.

2.  **Efficient Data Processing**: An initial analysis highlighted the importance of vectorized operations. Calculating token lengths using the `datasets.map()` function was over **14 times faster** than using a standard Python `for` loop (0.6 seconds vs. 8.5 seconds).

3.  **Tokenization**: A custom function was created to tokenize both the source (English) and target (Turkish) texts. Key parameters included a `max_length` of 128 and `padding='longest'` to ensure uniform batch shapes.

4.  **Training Configuration**: The `Seq2SeqTrainingArguments` were configured to optimize for speed and memory efficiency. Key settings included:
    * `per_device_train_batch_size=8` with `gradient_accumulation_steps=8` for an effective batch size of 64.
    * `fp16=True` to enable mixed-precision training for a significant speed-up.
    * `num_train_epochs=2`.

5.  **Evaluation**: The **SacreBLEU** metric from the `evaluate` library was used to measure translation quality. A `compute_metrics` function was implemented to decode model predictions and calculate the score during evaluation.

6.  **Model Training**: The `Seq2SeqTrainer` was used to handle the complete fine-tuning and evaluation loop.

7.  **Deployment**: After training, the model was successfully saved and uploaded to the Hugging Face Hub under the repository `gokhanErgul/kde4-en-to-tr_with-Helsinki`.

---
## 🛠️ How to Use the Fine-Tuned Model

You can easily use the fine-tuned model for your own English-to-Turkish translation tasks with just a few lines of code.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer from the Hugging Face Hub
model_path = "gokhanErgul/kde4-en-to-tr_with-Helsinki"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to(device)

# Prepare the text for translation
text_to_translate = "The book is on the table."
inputs = tokenizer(text_to_translate, return_tensors='pt')

# Move inputs to the selected device
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate the translation
output_tokens = model.generate(**inputs, max_length=128, num_beams=4)

# Decode the generated tokens into text
translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(f"Original: {text_to_translate}")
print(f"Translated: {translated_text}")

# Expected Output:
# Original: The book is on the table.
# Translated: Kitap masanın üzerinde.
