import os, random
import numpy as np
import torch

SEED = 4024                              # pick one number and re-use it everywhere
os.environ["PYTHONHASHSEED"] = str(SEED) # ① python string-hash determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # ② CuBLAS deterministic kernels

# ③ Pure-python RNGs
random.seed(SEED)
np.random.seed(SEED)

# ④ PyTorch RNGs (CPU + all visible GPUs)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ⑤ Make CuDNN / CuBLAS pick deterministic code paths
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False   # disable auto-tune