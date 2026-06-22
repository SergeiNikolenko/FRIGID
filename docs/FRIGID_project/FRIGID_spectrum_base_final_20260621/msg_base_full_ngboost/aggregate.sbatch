#!/bin/bash
#SBATCH --job-name=frigid-base-agg
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nikolenko/work/Projects/FRIGID/repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost/logs/aggregate-%j.out
#SBATCH --error=/home/nikolenko/work/Projects/FRIGID/repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost/logs/aggregate-%j.err
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
.venv/bin/python repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost/aggregate_base_ngboost.py
