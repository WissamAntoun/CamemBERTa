#!/bin/sh
#SBATCH --job-name=camembertav2-base-bf16-p2     # Name of job
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node (= number of GPUs per node)
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --hint=nomultithread         # hyperthreading deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=./logs/%x_%j.out # name of output file
#SBATCH --error=./logs/%x_%j.out  # name of error file (here, in common with the output file)
#SBATCH --exclusive


module purge
module load singularity

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "JOB ID            = $SLURM_JOB_ID"
echo ""
echo "Hostname                       = $SLURM_NODELIST"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of CPUs on host         = $SLURM_CPUS_ON_NODE"
echo "GPUs                           = $GPU_DEVICE_ORDINAL"

set -x

SINGULARITY_IMG=/singularity_sif/camemberta-tf2.16-ngc24.06.sif

CMD="cd /workspace/electra && ls && \
export DATA_PREP_WORKING_DIR=/scratch/camembertv2/data/ && \
export PYTHONUNBUFFERED=1 && \
export HOROVOD_ENABLE_XLA_OPS=1 && \
export TF_EXTRA_PTXAS_OPTIONS=\"-sw200428197=true\" && \
export TF_XLA_FLAGS=\"--tf_xla_auto_jit=fusible\" && \
nvidia-smi && nvidia-smi topo -m && printenv && \
pip freeze && \
./scripts/bind.sh --cpu=node --ib='' --cluster='' -- \
python run_pretraining.py --config_file ./configs/camembertav2_base_p2.json"

srun --mpi=pmix -l singularity exec \
-B "$PWD:/workspace/",\
"<SCRATCH>:/scratch" \
--nv \
$SINGULARITY_IMG \
bash -c "$CMD"