#!/bin/bash
python train.py --use_lora --max_epochs 30 --dataset "../llava_medical_multi_question_dataset" --lora_rank 8 --batch_size 1 --lr 2e-5 --strategy auto
