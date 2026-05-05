<p align="center">
  <h2 align="center">CT-AGRG: Automated Abnormality-Guided Report Generation for CT Scans 👨🏻‍⚕️📝</h2>
  <h4 align="center"><b>ISBI 2025</b></h4>
  <p align="center">
    <a href="https://arxiv.org/abs/2408.1196"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2408.1196-b31b1b.svg"></a>
  </p>
</p>

---

### ✨ Method Overview

CT-AGRG employs a two-stage approach for anomaly detection and description generation. **Pretraining.** Initially, a visual feature extractor is pre-trained on a multi-label classification task.  **Step1.** In the first stage, the visual encoder is fine-tuned with multi-task learning, with one classification head per anomaly.  **Step2.** If an anomaly is detected, its associated vector representation is then passed to the second stage. Here, a pre-trained GPT-2 model generates a descriptive text of the identified anomaly.

<img src="https://github.com/theodpzz/ct-agrg/blob/master/figures/method_overview.jpg" alt="Method overview" width="900">

---

### Notice

This repository is currently under review for compliance with institutional and collaborative agreements.
Public release of the code is temporarily restricted.

The repository will be made publicly available once the approval process is completed.

---

### 🙌 Acknowledgement

CT-AGRG implementations builds upon prior work, including [RGRG](https://github.com/ttanida/rgrg), [CT-Net](https://github.com/rachellea) and [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP). We thank the authors of these projects, as well as the contributors of [CT-RATE](https://arxiv.org/abs/2403.17834v3) for releasing the dataset to the research community.

---

### Purpose

This code is provided for **academic and research purposes only**, to support reproducibility of the results described in the associated paper. This repository is a research prototype, and is not intended for clinical use.

---

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
