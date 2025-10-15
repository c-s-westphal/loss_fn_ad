#!/bin/bash
# Master script to submit all VGG masked loss experiments
# Submits 4 job arrays (one per architecture) with 30s delays
# Total: 24 jobs (4 architectures × 6 jobs each)

echo "========================================"
echo "Submitting all VGG masked loss experiments"
echo "========================================"
echo ""

# Submit VGG11 jobs
echo "Submitting VGG11 jobs (6 tasks)..."
qsub scripts/job_manager_vgg11.sh
echo "Waiting 30 seconds before next submission..."
sleep 30
echo ""

# Submit VGG13 jobs
echo "Submitting VGG13 jobs (6 tasks)..."
qsub scripts/job_manager_vgg13.sh
echo "Waiting 30 seconds before next submission..."
sleep 30
echo ""

# Submit VGG16 jobs
echo "Submitting VGG16 jobs (6 tasks)..."
qsub scripts/job_manager_vgg16.sh
echo "Waiting 30 seconds before next submission..."
sleep 30
echo ""

# Submit VGG19 jobs
echo "Submitting VGG19 jobs (6 tasks)..."
qsub scripts/job_manager_vgg19.sh
echo ""

echo "========================================"
echo "All jobs submitted!"
echo "Total: 24 jobs (4 × 6)"
echo ""
echo "Monitor with: qstat"
echo "Cancel with: qdel <job_id>"
echo "========================================"
