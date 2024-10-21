#!/bin/bash
python train.py --use_qlora False --max_epochs 5 --dataset "../llava_medical_short_dataset" --lora_rank 8 -batch_size 1 --lr 2e-5
