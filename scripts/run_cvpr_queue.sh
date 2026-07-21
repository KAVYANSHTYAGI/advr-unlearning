#!/usr/bin/env bash
cd "$(dirname "$0")/.."
Q=results/seg/cvpr_queue.log
echo "[queue] START $(date)" > $Q
echo "[queue] === CITYSCAPES ===" >> $Q
bash scripts/run_cityscapes_experiment.sh >> $Q 2>&1 || echo "[queue] cityscapes FAILED" >> $Q
echo "[queue] === CAR ===" >> $Q
bash scripts/run_car_methods.sh >> $Q 2>&1 || echo "[queue] car FAILED" >> $Q
echo "[queue] === SEGFORMER ===" >> $Q
bash scripts/run_segformer_experiment.sh >> $Q 2>&1 || echo "[queue] segformer FAILED" >> $Q
echo "[queue] ALL DONE $(date)" >> $Q
