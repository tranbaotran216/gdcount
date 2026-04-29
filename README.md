---
title: GDCount
sdk: docker
app_port: 8501
suggested_hardware: t4-medium
---

# GDCount
Seminar project for CS331-UIT
This training is run on the GPU RTX 3060 12GB
## 1) Setup environment (Windows / VSCode Terminal)

### 1. Create conda env from YAML
you will need to check the cuda version in your pc to change the cuda version in file env.yaml
```powershell
conda env create -f env.yaml
conda activate gdcount
```

### 2. Install groundingdino
You need to export CUDA_HOME first. Detail can be found in [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO.git)
```bash
echo $env:CUDA_HOME (for powershell in windows). accessing GroundingDINO repo if you're using Linux/macOS.
```

**Note**: After cloning the gdcount repo, u must clone the official groundingdino repo (clone into this repo)
```powershell
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd ./GroundingDINO
pip install -e . --no-build-isolation
```

### 3. Run
Every code to run can be found in the file test_gd.ipynb

### 4. App
[checkpoints for all versions](https://drive.google.com/drive/folders/1jNEpp15Tcg04G4as1drDZ8MF1IQ9B_cc?usp=sharing) 
```powershell
streamlit run app.py
# or any app version 
```
or trying DEMO on HF [HERE](https://huggingface.co/spaces/Phiphi216/GDCount)

---
#Acknowledgements
The model is built following the paper [CountGD](https://github.com/niki-amini-naieni/CountGD)<br>
This repository is based on the Open-GroundingDino and uses code from the [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO.git)<br>
Thanks for your great work!<br>
@InProceedings{AminiNaieni24,
  author = "Amini-Naieni, N. and Han, T. and Zisserman, A.",
  title = "CountGD: Multi-Modal Open-World Counting",
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
  year = "2024",
}
