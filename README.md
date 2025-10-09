# PerAugy: Diversity Augmentation of Dynamic User Preference Data for Boosting Personalized Text Summarizers

This repository provides a step-by-step pipeline for preparing and processing the PENS dataset for sequential recommendation tasks using synthetic user interaction graphs (UIGs).

## 📦 Dataset

* **PENS Dataset**: Includes `train`, `validation`, `test`, and `news` sets.

---

## ⚙️ Processing Steps

### 1. Download the PENS Dataset

Start by downloading the PENS dataset, which contains the following files:

* `train.json`
* `val.json`
* `test.json`
* `news.json`

Ensure these files are placed in the appropriate directory for further processing.

---

### 2. Arrange Dataset into Seed UIG Format

Run the script:

```bash
python Scripts/PENS_augmentation.py
```

This script:

* Sorts user interactions by timestamp.
* Appends summary nodes from the test set.
* Generates a **Seed UIG** (user interaction graph) structure.

The following file will be created:

* `synthetic-original.csv` – the initial synthetic UIG dataset.

Additionally, a corresponding **summary dataset** will be generated with metadata about the summary nodes for each synthetic user.

---

### 3. Create Double Shuffled Trajectories

Open and run:

```bash
DS/DoubleShuffling.ipynb
```

This notebook generates **Double Shuffled (DS)** user trajectories. You can experiment with:

* `offset`
* `gap`
* `segment length`
* other relevant hyperparameters

---

### 4. Apply History-Influenced Perturbation

Run the following notebook to refine the DS dataset:

```bash
perturbation/perturbation_D2.ipynb
```

This step smooths the shuffled trajectories by applying history-aware perturbations to improve recommendation dynamics.

---

### 5. Convert to PENS Sequential Format

Run:

```bash
KG2PENS/KG2PENS_Trainer_New_Convertor.ipynb
```

This notebook:

* Replaces `<d-s>` pairs with corresponding `s-nodes` only.
* Converts the processed dataset into PENS-compatible knowledge graph format.

The final output format will be:

```
UserID, ClickedNewsID, PositiveNewsID, NegativeNewsID
```

---

## 📁 Output

After completing all steps, you’ll obtain:

* A PENS-style dataset ready for sequential recommendation experiments.
* Summary statistics and user graph structures aligned with experimental design.

---

## 📌 Notes

* Make sure all dependencies are installed before running the notebooks.
* Intermediate datasets are saved automatically after each step.
* You can tweak parameters in the Jupyter notebooks for different experimental settings.
