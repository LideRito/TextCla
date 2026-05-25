# Chinese News Text Classification

Multi-class text classification on Chinese news articles using Long BERT + CNN, built with [kashgari](https://github.com/BrikerMan/Kashgari).

> Originally developed as part of a team submission for the **National College Student Information Security Contest**. As a team member, I was responsible for this text classification module. The team was awarded the **National First Prize**. The code has been revisited and reorganized in 2026 as part of my internship application portfolio.

---

## Environment

- Python 3.6.13
- kashgari 2.0.1
- TensorFlow
- tensorflow-addons

```bash
pip install kashgari==2.0.1 tensorflow tensorflow-addons
```

Windows users also need `zlib-wapi`:

```bash
conda install -c conda-forge zlib-wapi
```

---

## Dataset

**CNNews** — 28,099 Chinese news articles across 10 categories.

| Split | Samples |
|-------|---------|
| Train | 21,220 |
| Validation | 2,258 |
| Test | 4,621 |

| Label | Category |
|-------|----------|
| 体育 | Sports |
| 娱乐 | Entertainment |
| 家居 | Home & Lifestyle |
| 房产 | Real Estate |
| 教育 | Education |
| 时尚 | Fashion |
| 时政 | Politics / Current Affairs |
| 游戏 | Gaming |
| 科技 | Technology |
| 财经 | Finance & Economics |

---

## Model

- **Embedding**: [Long BERT](https://huggingface.co/OctopusMind/longbert-embedding-8k-zh) (`longbert-embedding-8k-zh`), 768-dim, max sequence length 8,192
- **Classifier**: 1D-CNN (128 filters, kernel=5) → Dense (64) → Softmax (10 classes)
- **Training**: batch size 2, epochs 3, Adam optimizer

Using Long BERT instead of standard BERT extends the token limit from 512 to 8,192, allowing the model to handle full-length news articles without truncation.

---

## Usage

**Training**

1. Download the Long BERT model (PyTorch format) from [HuggingFace](https://huggingface.co/OctopusMind/longbert-embedding-8k-zh) and place it under `long-bert/`.

2. Convert the PyTorch checkpoint to TensorFlow format:

```bash
python bin2ckpt.py
```

This outputs the converted checkpoint to `long-bert/tf_ckpt/`.

3. Place the CNNews dataset under `cnews/`, then train:

```bash
python train.py
```

The trained model is saved to `./bert_emb_model/`.

**Inference**

```bash
python predict.py
```

Loads the trained model from `bert_emb_model/` and classifies input text interactively.

---

## Project Structure

```
textCla/
├── train.py          # Main training script
├── predict.py        # Interactive inference
├── bin2ckpt.py       # PyTorch → TensorFlow checkpoint converter
├── cnews/            # CNNews dataset
├── long-bert/        # Pre-trained Long BERT (download separately)
└── bert_emb_model/   # Trained model output
```

---

*This README was drafted with the assistance of [Claude Code](https://claude.ai/code). Technical content was reviewed and verified by the author.*
