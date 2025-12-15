original, follows the pipeline of countgd repo

differences're not conducted

## 1) Setup environment (Windows / VSCode Terminal)

### 1.1 Create conda env từ YAML
```powershell
conda env create -f env.yaml
conda activate gdcount
```

### 1.2 Install groundingdino
```powershell
cd ./groundingdino
pip install -e .
```

### Run
every code to run can be found in file test_gd.ipynb
