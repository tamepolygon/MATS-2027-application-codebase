#!/bin/bash
#SBATCH --job-name=em-envcheck
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

source sbatch/_preamble.sh

echo "--- torch / device ---"
python -c "import torch;print('torch',torch.__version__,'| built for CUDA',torch.version.cuda,'| archs',torch.cuda.get_arch_list(),'| device',torch.cuda.get_device_name(0),'| cc',torch.cuda.get_device_capability(0),'| bf16',torch.cuda.is_bf16_supported());import transformers,peft;print('transformers',transformers.__version__,'| peft',peft.__version__)"

echo "--- bf16 matmul actually runs (this is what fails when sm_90 kernels are missing) ---"
python -c "import torch;x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16);print('result',(x@x).float().mean().item());print('peak VRAM GB',torch.cuda.max_memory_allocated()/1e9)"

echo "--- cache contents ---"
python src/fetch_models.py --cache "$EM_CACHE" --verify-only

echo "--- eval assets parse ---"
python src/eval_assets.py

echo "ENV CHECK COMPLETE"
