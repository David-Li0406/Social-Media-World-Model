#!/bin/bash
#SBATCH --job-name=buildTrainTest
#SBATCH -A grp_huanliu
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/widthSummarize_%j.out
#SBATCH --error=logs/widthSummarize_%j.err

cd /scratch/akutteti/reddit_dump_analysis
mkdir -p logs

module load mamba/latest
source activate lf

/home/akutteti/.conda/envs/lf/bin/python3 /scratch/akutteti/reddit_dump_analysis/buildTrainTest.py