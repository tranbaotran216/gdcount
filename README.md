---
title: GDCount
sdk: docker
app_port: 8501
suggested_hardware: t4-medium
---

# GDCount: Image-Adaptive Anti-Hallucination Zero-Shot Object Counting

> **Seminar Project CS331 - Computer Vision (UIT VNU-HCM)**  
> **Experimental Environment:** Trained on an NVIDIA GeForce RTX 3060 12GB GPU  
> **Recommended Checkpoint:** `gdcount_epoch_013_best.pth` (Calibrated Model)

---

## 🌟 Overview & Key Innovations

GDCount is an open-world Zero-Shot Object Counting (ZSC) framework built upon the **GroundingDINO** foundation detector (Swin Transformer + BERT), enhanced with a custom **Small Object Adapter (SOA)** and a **ScoreCalibrator** head. 

In this upgraded release, the counting pipeline is augmented with a post-processing **Image-Adaptive Prototype Refinement and Anti-Hallucination Verification Framework**. This enhanced post-processing operates strictly at inference time with zero backbone fine-tuning, successfully resolving the two most prominent failure modes in open-world counting:

### 1. Target-Absence Hallucination Mitigation (Object-Presence Hallucination - OPH)
* **The Failure Mode:** Contemporary zero-shot counters operate under an implicit *Positive-Query Assumption*—presuming the queried category is invariably present in the scene. When queried with an object category that does not exist in the image (e.g., querying *"elephant"* on an image containing only *"strawberries"*), standard detectors forcibly project bounding boxes onto salient distractors or background textures, yielding spurious non-zero counts.
* **The Proposed Enhancement:**
  - Integrates an explicit binary **Presence Gate**: $\hat{z} \in \{0, 1\}$.
  - Constructs a query-conditioned candidate *Seed Pool* and enforces sequential consistency ($\tau_c = 0.70$).
  - If the target concept is absent or fails to establish a consistent visual seed pool, all candidate proposals are suppressed, rigorously enforcing:
    $$P(\hat{C} = 0 \mid \text{Target Absent}) = 1$$

### 2. Multi-Object Co-Presence Disambiguation
* **The Failure Mode:** In cluttered real-world scenes displaying multiple object categories simultaneously (e.g., a table with *apples*, *oranges*, and *bottles*), baseline counters frequently misidentify visually similar distractors and over-count cross-category objects.
* **The Proposed Enhancement:**
  - Employs an offline-indexed **Competitor Feature Bank** spanning 147 FSC-147 categories.
  - Implements competitive argmax class attribution:
    $$c^* = \arg\max_{c \in \mathcal{C}} \text{Sim}(v_i, c)$$
  - Candidates whose visual representations align more strongly with an alternative co-present category ($c^* \neq q$) are pruned as distractors.

### 3. Pure Post-Processing Image-Adaptive Prototype ($p_{\text{adapt}}$)
The enhancement operates strictly at inference time as a modular post-processor, requiring no retraining of the detector backbone:
1. Extract L2-normalized ROI visual embeddings $v_i \in \mathbb{R}^D$ from candidate proposals using dense foundation features.
2. Dynamically aggregate high-confidence *trusted seeds* within the test image into an image-specific **Real Prototype ($p_{\text{real}}$)** that captures scene-level illumination, scale, and perspective.
3. Blend $p_{\text{real}}$ with the canonical synthetic prototype $p_{\text{syn}}^*$ ($\lambda = 0.20$):
   $$p_{\text{adapt}} = \text{normalize}\left((1 - \lambda) p_{\text{real}} + \lambda p_{\text{syn}}^*\right)$$
4. Filter out candidates failing the adapted cosine similarity threshold: $s_{\text{adapt}, i} = v_i^T p_{\text{adapt}} \ge \tau_{\text{adapt}} = 0.60$.

```mermaid
flowchart TD
    Img["Input Image I"] --> GD["GDCount Detector (Epoch 13 Calib)"]
    Prompt["Query Text q (e.g., 'strawberries')"] --> GD
    
    subgraph DETECTOR ["1. Candidate Proposal Generation"]
        GD --> RawBoxes["N Candidate Boxes + Calibrated Scores (g_i)"]
    end
    
    subgraph ADAPTIVE_POSTPROCESS ["2. Image-Adaptive Prototype Verification"]
        RawBoxes --> FeatExt["Extract Visual Embeddings: v_i (D = 1024) via DINOv3"]
        FeatExt --> PresGate{"Presence Gate:<br/>Target present in image?"}
        
        PresGate -->|"NO (Target Absent)"| Suppress["SUPPRESS ALL BOXES<br/>(Anti-Hallucination: Count = 0)"]
        
        PresGate -->|"YES (Target Present)"| SeedPool["Select In-Image Trusted Seeds"]
        SeedPool --> PReal["Construct Real-Image Prototype: p_real"]
        PReal --> PAdapt["Blend: p_adapt = 0.8 p_real + 0.2 p_syn"]
        
        PAdapt --> CompCheck{"Competitor Class Check:<br/>argmax Sim(v_i, c) == q ?"}
        CompCheck -->|"Mismatch (Belongs to other class)"| RejectComp["Reject Box (Co-present distractor)"]
        CompCheck -->|"Match & Sim >= 0.60"| KeepBox["Keep Valid Box"]
    end
    
    KeepBox --> FinalCount["Verified Object Count (Final Count)"]
    Suppress --> FinalCount
```

---

## 🎨 Role of Synthetic Images & Bridging the Domain Gap

### Why Synthetic Images?
Traditional Few-Shot Object Counting (e.g., FSC-147 baselines) requires users to manually annotate 3 bounding box exemplars for every test image. In a truly autonomous Zero-Shot Counting setting, human-in-the-loop annotations are unavailable.

To bridge this gap without human intervention:
1. **Diffusion-Based Exemplar Generation:** We use a high-capacity diffusion model (**FLUX.1 [dev]**) to synthesize canonical visual exemplar images for all 147 vocabulary categories (`syn_FSC147_classes`).
2. **Canonical Prototype Extraction:** For each synthetic image, standard exemplar bounding boxes are extracted and encoded into normalized canonical prototype vectors ($p_{\text{syn}}^*$) stored in the prototype bank.

### The Synthetic-to-Real Domain Gap
While synthetic exemplars provide clean, studio-lit representations of canonical shapes:
* Synthetic images often exhibit uniform backgrounds, ideal lighting, and isolated viewpoints.
* Real-world test images contain complex textures, shadows, occlusions, perspective distortion, and domain shift.

Directly matching real test proposals against static synthetic prototypes can cause severe under-counting. The **Image-Adaptive Prototype Refinement** solves this by dynamically creating an in-image prototype ($p_{\text{real}}$) from the highest-confidence instances found in the test image itself, then lightly anchoring it to the synthetic canon ($p_{\text{syn}}^*$). This eliminates the domain gap entirely at zero additional training cost.

---

## 🦖 DINOv3 Feature Extractor & Weights Setup

The adaptive verification module utilizes dense patch features from **DINOv3 ViT-L/16** (pre-trained with self-supervised visual representation learning) to perform high-resolution ROI feature extraction.

### 1. DINOv3 Dependencies
DINOv3 requires `torch`, `torchvision`, `timm`, and `einops`. These packages are included in the environment setup:
```powershell
pip install timm einops
```

### 2. DINOv3 Repository & Pretrained Checkpoint
1. Clone or locate the local DINOv3 repository:
   ```powershell
   # If cloning from official repository
   git clone https://github.com/facebookresearch/dinov3.git
   ```
2. Download the pre-trained ViT-L/16 checkpoint (`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` ~1.21 GB) and place it under your weights directory, for example:
   ```text
   dinov3/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
   ```
3. Set the environment paths or pass them via CLI/UI:
   - `DINOV3_REPO`: Path to the root DINOv3 directory (e.g. `D:\projects\zsc\dinov3`).
   - `DINOV3_WEIGHTS`: Path to the `.pth` weights file.

### 3. Precomputed Prototype Cache (`.dinov3_cache`)
The 147-class prototype bank is stored in `data/syn_FSC147_classes/.dinov3_cache/`:
* `prototypes.pt`: Tensor dictionary containing `prototype_matrix` of shape `[162, 1024]` and associated `class_keys`.
* `records.json`: Metadata for each synthetic prototype image, canonical bounding box coordinates, and source info.

*(Note: If DINOv3 weights are not downloaded, the codebase automatically falls back to an internal lightweight ROI patch embedding extractor, ensuring inference can run on any hardware without crashing).*

---

## 🛠️ Installation & Environment Setup

### 1. Create Conda Environment from YAML
```powershell
conda env create -f env.yaml
conda activate gdcount
```

### 2. Install GroundingDINO
Ensure the `CUDA_HOME` environment variable is set prior to compilation:
```powershell
# Windows PowerShell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"  # Adjust to your installed CUDA toolkit version
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd ./GroundingDINO
pip install -e . --no-build-isolation
cd ..
```

---

## 🚀 Running Inference & Applications

### 1. Interactive Streamlit Web App
Launch the comprehensive Streamlit demo featuring interactive post-processing toggles, color-coded bounding box attribution (Target vs. Competitor vs. Suppressed), and Gaussian density map overlays:
```powershell
streamlit run app_exp3.py
```
*(Or launch the lightweight baseline interface: `streamlit run app.py`)*

* **Live Demo on Hugging Face Spaces:** [Phiphi216/GDCount](https://huggingface.co/spaces/Phiphi216/GDCount)

### 2. Offline Inference Script
Run single-image inference directly from the command line:
```powershell
python test_exp3.py --image "CountGD/img/306.jpg" --prompt "strawberries" --threshold 0.23 --adapt_threshold 0.60
```

---

## 📦 Checkpoints & Pretrained Weights

* **Download Checkpoints:** [Google Drive Folder](https://drive.google.com/drive/folders/1jNEpp15Tcg04G4as1drDZ8MF1IQ9B_cc?usp=sharing)
  * `gdcount_epoch_013_best.pth`: Best calibrated model checkpoint (recommended for enhanced adaptive counting).
  * `gdcount_epoch_011_best_fp16.pth`: Lightweight FP16 quantized model for web deployment.
  * `groundingdino_swint_ogc.pth`: Pretrained backbone weights from IDEA-Research.

---

## 🙏 Acknowledgements

This project builds upon the foundational research and codebases of:
* [CountGD](https://github.com/niki-amini-naieni/CountGD) (NeurIPS 2024 - Amini-Naieni et al.)
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO.git) (IDEA-Research)
* DINOv3 (Meta AI Research)
* Research on Object-Presence Hallucination (OPH) & the PrACo Benchmark (Tran, Ninh, Dung - UIT VNU-HCM, 2026).
