# SGE Job Scripts for VGG Masked Loss Experiments

This directory contains job submission scripts for running VGG experiments on a Sun Grid Engine (SGE) cluster.

## Files

### Job Manager Scripts (executable)
- `job_manager_vgg11.sh` - VGG11 training jobs
- `job_manager_vgg13.sh` - VGG13 training jobs
- `job_manager_vgg16.sh` - VGG16 training jobs
- `job_manager_vgg19.sh` - VGG19 training jobs

### Parameter Files
- `jobs_vgg11.txt` - Parameters for VGG11 (6 tasks)
- `jobs_vgg13.txt` - Parameters for VGG13 (6 tasks)
- `jobs_vgg16.txt` - Parameters for VGG16 (6 tasks)
- `jobs_vgg19.txt` - Parameters for VGG19 (6 tasks)

Format: `mode seed` (one per line)
- modes: `baseline`, `masked`
- seeds: `0`, `1`, `2`

### Submission Scripts
- `submit_all.sh` - Submit all 24 jobs at once

## Usage

### Submit All Experiments (Recommended)

```bash
./scripts/submit_all.sh
```

This submits 4 job arrays (24 total jobs):
- VGG11: 6 jobs (2 modes × 3 seeds)
- VGG13: 6 jobs (2 modes × 3 seeds)
- VGG16: 6 jobs (2 modes × 3 seeds)
- VGG19: 6 jobs (2 modes × 3 seeds)

### Submit Individual Architectures

```bash
# Submit only VGG11
qsub scripts/job_manager_vgg11.sh

# Submit only VGG16
qsub scripts/job_manager_vgg16.sh
```

### Monitor Jobs

```bash
# Check job status
qstat

# Check specific user's jobs
qstat -u $USER

# Watch jobs in real-time
watch -n 5 qstat
```

### Cancel Jobs

```bash
# Cancel specific job
qdel <job_id>

# Cancel all your jobs
qdel -u $USER

# Cancel specific job array task
qdel <job_id>.<task_id>
```

## Job Configuration

Each job requests:
- **Memory**: 16GB (`#$ -l tmem=16G`)
- **Time**: 24 hours (`#$ -l h_rt=24:00:00`)
- **GPU**: Required (`#$ -l gpu=true`)
- **Tasks**: 6 per architecture (`#$ -t 1-6`)

## Output Files

Job output is saved to the current working directory:
- `<job_name>.o<job_id>.<task_id>` - stdout/stderr combined

Results are saved to:
- `checkpoints/` - Model checkpoints
- `results/` - Training histories (JSON)
- `data/` - CIFAR-10 dataset (auto-downloaded)

## Training Configuration

Each job runs `train.py` with:
- Epochs: 200
- Batch size: 128
- Learning rate: 0.1 (SGD with momentum)
- Weight decay: 5e-4
- Number of masks: 3 (for masked mode)
- Lambda max: 1.0 (for masked mode)

## After Training

Once all jobs complete, generate plots:

```bash
python visualize.py
```

## Troubleshooting

### Job fails immediately
- Check virtual environment path in job scripts
- Verify CUDA module paths
- Ensure `train.py` is in the root directory

### Out of memory
- Reduce batch size: `--batch_size 64`
- Request more memory: `#$ -l tmem=32G`

### Job times out
- Increase time limit: `#$ -l h_rt=48:00:00`
- Reduce epochs: `--epochs 100`

### Dataset download issues
- Pre-download CIFAR-10: `python -c "import torchvision; torchvision.datasets.CIFAR10('./data', download=True)"`

## Customization

To modify training parameters, edit the `python3.9 -u train.py` command in each job manager script.

To add more seeds, append to the jobs files:
```bash
echo "baseline 3" >> scripts/jobs_vgg11.txt
echo "masked 3" >> scripts/jobs_vgg11.txt
```

Then update the task array: `#$ -t 1-8` (for 8 tasks)
