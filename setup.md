# TurboVLA Setup (LIBERO Evaluation)

## 1. Clone the repo
```bash
git clone https://github.com/Sujay-Pravin/TurboVLA.git
cd TurboVLA
```

## 2. Create Python 3.10 environment
```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
```

## 3. Install PyTorch (CUDA 12.1)
```bash
UV_HTTP_TIMEOUT=300 uv pip install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install TurboVLA
```bash
uv pip install -e ".[libero]"
```

## 5. Clone and install LIBERO
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO/
touch ~/projects/TurboVLA/LIBERO/libero/__init__.py
uv pip install -e .
cd ../
```

## 6. Download pretrained models
```bash
mkdir -p pretrained_models

# BERT text encoder
huggingface-cli download google-bert/bert-base-uncased \
    --local-dir pretrained_models/bert-base-uncased

# DINOv3 vision encoder
huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m \
    --local-dir pretrained_models/dinov3-vitb
```

## 7. Install GroundingDINO + weights
```bash
cd pretrained_models/
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/

python -m ensurepip --upgrade
python -m pip install -e .

mkdir weights
cd weights/
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../
```

## 8. Generate LIBERO instructions
```bash
python -c '
from libero.libero import benchmark
b_dict = benchmark.get_benchmark_dict()
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
instructions = set()
for s in suites:
    suite = b_dict[s]()
    for i in range(suite.n_tasks):
        instructions.add(suite.get_task(i).language)
with open("data/libero_instructions.txt", "w") as f:
    for inst in sorted(instructions):
        f.write(inst + "\n")
print(f"Extracted {len(instructions)} unique LIBERO instructions.")
'
```

## 9. Build the text cache
```bash
# Uses the locally downloaded BERT model as the text encoder
python scripts/libero/build_text_cache.py \
    --text_encoder_type ./pretrained_models/bert-base-uncased \
    --instructions_file data/libero_instructions.txt \
    --output data/libero_all4_bert_text_cache.pt \
    --overwrite
```

## 10. Download TurboVLA checkpoints
```bash
hf download H-EmbodVis/TurboVLA --local-dir pretrained_models/TurboVLA
```

## 11. Install system rendering dependencies
```bash
sudo apt update
sudo apt install \
    libosmesa6 \
    libosmesa6-dev \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libegl1 \
    libglew-dev \
    patchelf
```

## 12. Configure OSMesa rendering env vars
```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Sanity check
echo $MUJOCO_GL
echo $PYOPENGL_PLATFORM
```

## 13. Install compatible MuJoCo + PyOpenGL
```bash
uv pip install "PyOpenGL==3.1.7" "PyOpenGL-accelerate==3.1.7"

uv pip uninstall mujoco
uv pip install mujoco==2.3.7
```

## 14. Verify setup
```bash
python -c "import mujoco; print(mujoco.__version__)"   # expect: 2.3.7
python -c "import torch; print(torch.cuda.is_available())"  # expect: True
```

## 15. Run evaluation
```bash
# Use --mujoco-gl egl for GPU-accelerated rendering (faster than osmesa)
CUDA_VISIBLE_DEVICES=0 python experiments/libero/evaluate.py \
    --ckpt_path pretrained_models/TurboVLA/checkpoints/libero/object.pth \
    --dinov3_path pretrained_models/dinov3-vitb \
    --text_cache_path data/libero_all4_bert_text_cache.pt \
    --stats_path experiments/libero/configs/libero_all4_stats.json \
    --stats_key libero_all4_no_noops \
    --task_suite_name libero_object \
    --num_trials_per_task 50 \
    --chunk_size 12 \
    --num_open_loop_steps 12 \
    --precision bf16 \
    --mujoco-gl egl \
    --result_json_path outputs/evaluation/libero_object.json

#Use in CPU
    PYOPENGL_PLATFORM="osmesa" 
    CUDA_VISIBLE_DEVICES="" 
    python experiments/libero/evaluate.py     
    --ckpt_path pretrained_models/TurboVLA/checkpoints/libero/object.pth     
    --dinov3_path pretrained_models/dinov3-vitb     
    --text_cache_path data/libero_all4_bert_text_cache.pt     
    --stats_path experiments/libero/configs/libero_all4_stats.json     
    --stats_key libero_all4_no_noops     
    --task_suite_name libero_object     
    --num_trials_per_task 50     
    --chunk_size 12     
    --num_open_loop_steps 12    
    --precision fp32    
    --mujoco-gl osmesa  
    --result_json_path outputs/evaluation/libero_object.json
```

Results are saved to `outputs/evaluation/libero_object.json` (per-task and overall success rates).

### Notes
- `datasets path .../LIBERO/datasets does not exist!` — expected, harmless.
- `No private macro file found!` (Robosuite) — informational, ignore.
- Gym deprecation warning — informational, ignore.
