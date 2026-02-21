<p align="center">
  <h2 align="center">[ISBI 2025] CT-AGRG: Automated Abnormality-Guided Report Generation for CT Scans 👨🏻‍⚕️📝</h2>
</p>

✅ PyTorch implementation of "CT-AGRG: Automated Abnormality-Guided Report Generation for CT Scans".

📄 Accepted at ISBI 2025: [arXiv preprint](https://arxiv.org/abs/2408.11965).

---

### ✨ Method Overview

CT-AGRG employs a two-stage approach for anomaly detection and description generation. **Pretraining.** Initially, a visual feature extractor is pre-trained on a multi-label classification task.  **Step1.** In the first stage, the visual encoder is fine-tuned with multi-task learning, with one classification head per anomaly.  **Step2.** If an anomaly is detected, its associated vector representation is then passed to the second stage. Here, a pre-trained GPT-2 model generates a descriptive text of the identified anomaly.

<img src="https://github.com/theodpzz/ct-agrg/blob/master/figures/method_overview.jpg" alt="Method overview" width="900">

### 🚀 Getting Started

#### 1. Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/theodpzz/ct-agrg.git
```

#### 2. Installation

Make sure you have Python 3 installed. Then, install the dependencies using:

```bash
pip install -r requirements.txt
```
#### 3. Requirements

For step 2 (report generation), download the pretrained GPT-2 weights from: 🤗 [https://huggingface.co/healx/gpt-2-pubmed-medium](https://huggingface.co/healx/gpt-2-pubmed-medium).

#### 4. Dataset

CT-AGRG was trained and evaluated on the [CT-RATE dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

**Important**: For each abnormality, the corresponding sentence must be extracted from the report. This can be done using the RadBERT labeler provided in the original CT-RATE release.

#### 5. Training

Training is organized per step. Please, navigate to the corresponding folder and follow the provided instructions inside each directory.

#### 6. Demo

An inference notebook is available at **./notebook/demo.ipynb**.

### ⚙️ CT Scan Processing

CT scans are reformated such that the first axis points from Inferior to Superior, the second from Right to Left, and the third from Anterior to Posterior (SLP). The voxel spacing is (z, x, y) = (1.5, 0.75, 0.75) millimeters. The Hounsfield Units are clipped to [-1000, +200], and mapped to the range [0, 1] before normalization using ImageNet statistic (-0.449).

<img src="https://github.com/theodpzz/ct-scroll/blob/master/figures/orientation.png" alt="Orientation" width="900">

### 🔥 Available resources

CT-AGRG pretrained weights on CT-RATE are available at 🤗 [https://huggingface.co/theodpzz/ct-agrg](https://huggingface.co/theodpzz/ct-agrg).

### 🙌 Acknowledgement

CT-AGRG implementations builds upon prior work, including [RGRG](https://github.com/ttanida/rgrg), [CT-Net](https://github.com/rachellea) and [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP). We thank the authors of these projects, as well as the contributors of [CT-RATE](https://arxiv.org/abs/2403.17834v3) for releasing the dataset to the research community.

## 📎Citation

If you use this repository in your work, we would appreciate the following citation:

```bibtex
@InProceedings{dipiazza_2025_ctagrg,
        title = {CT-AGRG: Automated Abnormality-Guided Report Generation for CT Scans},
      	author = {Di Piazza, Theo and Lazarus, Carole and Nempont, Olivier and Boussel, Loic},
      	booktitle = {2025 {IEEE} 22nd {International} {Symposium} on {Biomedical} {Imaging} ({ISBI})},
      	year = {2025},
}
```
