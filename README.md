# Thyroid-llava-next


## Login to wandb and hugigngface
```
huggingface-cli login
# paste: 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
wandb login 
# paste: f85958b3fee33286e7f323fafa1d4364c0cd5a22
```

## Create env
```
git clone https://github.com/Trgtuan10/Thyroid-llava-next.git
cd Thyroid-llava-next
pip install -r requirements.txt
```

## Create dataset_dict
```
python make_dataset_dict.py --data_dir "../images" --output_dir "../llava_medical_short_dataset"
# --data_dir is folder saving Thyroid image
```

## Train
```
python train.py --use_qlora False --max_epochs 5 --dataset "../llava_medical_short_dataset" --lora_rank 8 --batch_size 1 --lr 2e-5
```
