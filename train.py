from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from datasets import DatasetDict
import torch
import datetime
from dataset import LlavaNextDataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from torch.utils.data import DataLoader
from nltk import edit_distance
import numpy as np
import argparse
from typing import List

from evaluation import (
    extract_TIRADS,
    extract_results,
    extract_size,
    extract_position,
    average_precision,
    calculate_iou_size,
    extract_all,
    extract_bbox,
)




# Define LlavaModelPLModule class here (same as before)
class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset, train_collate_fn, eval_collate_fn):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn

        self.batch_size = config.get("batch_size")

        self.validation_outputs = []

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, image_sizes, labels = batch

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             pixel_values=pixel_values,
                             image_sizes=image_sizes,
                             labels=labels
                             )
        loss = outputs.loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # Overwrite validation_step to implement custom validation logic
        input_ids, attention_mask, pixel_values, image_sizes, answers, question_ids = batch

        # Generate predictions using autoregression
        generated_ids = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            pixel_values=pixel_values.to(self.device),
            image_sizes=image_sizes,
            max_new_tokens=self.config.get("max_new_tokens", 128)
        )

        # Decode the generated tokens into text
        predictions = self.processor.batch_decode(
            generated_ids[:, input_ids.size(1):], skip_special_tokens=True
        )

        # Collect outputs
        outputs = []
        for pred, answer, question_id in zip(predictions, answers, question_ids):
            outputs.append({
                "question_id": int(question_id),
                "answer": answer,
                "model_answer": pred
            })

        # Store outputs for later use in on_validation_epoch_end
        self.validation_outputs.extend(outputs)
        print("outputs", outputs)   

    def on_validation_epoch_end(self):
        # This method is called automatically at the end of the validation epoch
        # Process all collected outputs
        all_outputs = self.validation_outputs

        # Reset the validation outputs for the next epoch
        self.validation_outputs = []

        # Initialize scores and counts
        scores = {
        'acc_tirads': 0,
        'acc_results': 0,
        'iou_size': 0,
        'iou_bbox': 0,
        'acc_position': 0
        }

        sample_counts = {
            'tirads': 0,
            'results': 0,
            'size': 0,
            'bbox': 0,
            'position': 0
        }

        not_found_counts = {
            'tirads': 0,
            'results': 0,
            'size': 0,
            'bbox': 0,
            'position': 0
        }
        all_detected_boxes = []
        all_gt_boxes = []

        for example in all_outputs:
            question_id = int(example.get("question_id"))
            answer = example.get("answer")
            model_answer = example.get("model_answer")

            # Include your evaluation logic here

            if question_id == 0:
                # TIRADS
                model_tirads = extract_TIRADS(model_answer)
                true_tirads = extract_TIRADS(answer)
                if model_tirads != "Not Found" and true_tirads != "Not Found":
                    if model_tirads == true_tirads:
                        scores['acc_tirads'] += 1
                    sample_counts['tirads'] += 1
                else:
                    not_found_counts['tirads'] += 1

            elif question_id == 1:
                # Size
                model_size = extract_size(model_answer)
                true_size = extract_size(answer)
                if model_size and true_size:
                    iou_size = calculate_iou_size(model_size, true_size)
                    scores['iou_size'] += iou_size
                    sample_counts['size'] += 1
                else:
                    not_found_counts['size'] += 1

                # Position
                model_position = extract_position(model_answer)
                true_position = extract_position(answer)
                if model_position != "Not Found" and true_position != "Not Found":
                    if model_position == true_position:
                        scores['acc_position'] += 1
                    sample_counts['position'] += 1
                else:
                    not_found_counts['position'] += 1

            elif question_id == 2:
                # Bbox
                model_bbox = extract_bbox(model_answer)
                true_bbox = extract_bbox(answer)
                if model_bbox['Bbox'] != "Not Found" and true_bbox['Bbox'] != "Not Found":
                    # iou_bbox = calculate_iou_bbox(model_bbox['Bbox'], true_bbox['Bbox'])
                    # scores['iou_bbox'] += iou_bbox
                    # sample_counts['bbox'] += 1
                    confidences = 1.0
                    all_detected_boxes.append((model_bbox['Bbox'], confidences))
                    all_gt_boxes.append(true_bbox['Bbox'])

                else:
                    not_found_counts['bbox'] += 1

            elif question_id == 3:
                # Results
                model_results = extract_results(model_answer)
                true_results = extract_results(answer)
                if model_results != "Not Found" and true_results != "Not Found":
                    if model_results == true_results:
                        scores['acc_results'] += 1
                    sample_counts['results'] += 1
                else:
                    not_found_counts['results'] += 1

            elif question_id == 4:
                # All parts
                model_data = extract_all(model_answer)
                true_data = extract_all(answer)

                # TIRADS
                if model_data["TIRADS"] != "Not Found" and true_data["TIRADS"] != "Not Found":
                    if model_data["TIRADS"] == true_data["TIRADS"]:
                        scores['acc_tirads'] += 1
                    sample_counts['tirads'] += 1
                else:
                    not_found_counts['tirads'] += 1

                # Results
                if model_data["Results"] != "Not Found" and true_data["Results"] != "Not Found":
                    if model_data["Results"] == true_data["Results"]:
                        scores['acc_results'] += 1
                    sample_counts['results'] += 1
                else:
                    not_found_counts['results'] += 1

                # Size
                if model_data["Size"] != "Not Found" and true_data["Size"] != "Not Found":
                    iou_size = calculate_iou_size(model_data["Size"], true_data["Size"])
                    scores['iou_size'] += iou_size
                    sample_counts['size'] += 1
                else:
                    not_found_counts['size'] += 1

                # Bbox
                if model_data["Bbox"] != "Not Found" and true_data["Bbox"] != "Not Found":
                    # iou_bbox = calculate_iou_bbox(model_data["Bbox"], true_data["Bbox"])
                    # scores['iou_bbox'] += iou_bbox
                    # sample_counts['bbox'] += 1
                    confidences = 1.0
                    all_detected_boxes.append((model_bbox['Bbox'], confidences))
                    all_gt_boxes.append(true_bbox['Bbox'])
                else:
                    not_found_counts['bbox'] += 1

                # Position
                if model_data["Position"] != "Not Found" and true_data["Position"] != "Not Found":
                    if model_data["Position"] == true_data["Position"]:
                        scores['acc_position'] += 1
                    sample_counts['position'] += 1
                else:
                    not_found_counts['position'] += 1

        # Calculate average scores
        avg_acc_tirads = scores['acc_tirads'] / sample_counts['tirads'] if sample_counts['tirads'] > 0 else 0
        avg_acc_results = scores['acc_results'] / sample_counts['results'] if sample_counts['results'] > 0 else 0
        avg_iou_size = scores['iou_size'] / sample_counts['size'] if sample_counts['size'] > 0 else 0
        avg_acc_position = scores['acc_position'] / sample_counts['position'] if sample_counts['position'] > 0 else 0

        # Compute mAP if there are bbox samples
        if all_detected_boxes:
            ap_50 = average_precision(all_detected_boxes, all_gt_boxes, iou_threshold=0.5)
            ap_75 = average_precision(all_detected_boxes, all_gt_boxes, iou_threshold=0.75)
            ap_95 = average_precision(all_detected_boxes, all_gt_boxes, iou_threshold=0.95)
        else:
            ap_50 = ap_75 = ap_95 = 0.0

        # Log the metrics
        self.log("val_acc_tirads", avg_acc_tirads, prog_bar=True, logger=True)
        self.log("val_iou_size", avg_iou_size, prog_bar=True, logger=True)
        self.log("val_ap_50", ap_50, prog_bar=True, logger=True)
        self.log("val_ap_75", ap_75, prog_bar=True, logger=True)
        self.log("val_ap_95", ap_95, prog_bar=True, logger=True)
        self.log("val_acc_results", avg_acc_results, prog_bar=True, logger=True)
        self.log("val_acc_position", avg_acc_position, prog_bar=True, logger=True)

        #log not found
        self.log("z_val_not_found_tirads", not_found_counts['tirads'], prog_bar=True, logger=True)
        self.log("z_val_not_found_results", not_found_counts['results'], prog_bar=True, logger=True)
        self.log("z_val_not_found_size", not_found_counts['size'], prog_bar=True, logger=True)
        self.log("z_val_not_found_position", not_found_counts['position'], prog_bar=True, logger=True)

   

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=self.eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def train_collate_fn(examples):
    images = []
    texts = []

    for example in examples:
        question = example["question"]
        image = example["image"]
        answer = example["answer"]

        # Add the image to the batch
        images.append(image)

        # Prepare the conversation (with the image token in the text)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # This is where you refer to the image token
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            }
        ]

        # Generate text prompt, ensuring the image token is added
        text_prompt = processor.apply_chat_template(conversation)
        texts.append(text_prompt)

    # Process the batch (tokenizing both text and image data)
    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    # Prepare the labels for training (mask padding tokens)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    # Return the necessary tensors
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_sizes"], batch["labels"]

def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []
    question_ids = []

    # Loop through the examples in the batch
    for example in examples:
        question = example["question"]       # Extract the user's question
        image = example["image"]             # Extract the image
        answer = example["answer"]           # Extract the assistant's answer (text)
        question_id = example["question_id"] # Extract the question ID

        # Add image to the batch
        images.append(image)

        # Create conversation structure (without the assistant's answer for evaluation)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Reference to the image
                    {"type": "text", "text": question},  # User's question
                ],
            }
        ]

        # Apply the conversation template with generation prompt enabled for evaluation
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        texts.append(text_prompt)

        # Save the answer and question ID to compare later
        answers.append(answer)
        question_ids.append(question_id)

    # Process text and images together
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Return the necessary tensors for the model's input, along with ground truth answers and question IDs for evaluation
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_sizes"], answers, question_ids


class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")

if __name__ == "__main__":
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA training.")
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA training.")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--val_check_interval", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPUs to use.")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--result_path", type=str, default="./result")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--dataset", type=str, default="../llava_medical_short_dataset")
    parser.add_argument("--percent", type=float, default=1)

    args = parser.parse_args()

    # Update config with parsed arguments
    config = {
        "max_epochs": args.max_epochs,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_nodes": args.num_nodes,
        "warmup_steps": args.warmup_steps,
        "result_path": args.result_path,
        "verbose": args.verbose,
    }

    MAX_LENGTH = 256
    MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
    # REPO_ID = "TrgTuan10/Thyroid-llava-next-multi-prompt"
    REPO_ID = "TrgTuan10/testing"
    WANDB_PROJECT = "LLaVaNeXT-Thyroid"

    now = datetime.datetime.now()
    WANDB_NAME = "thyroid-" + now.strftime("%Y-%m-%d-%H-%M-%S")

    USE_LORA = args.use_lora
    USE_QLORA = args.use_qlora

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    ## Load model
    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            # quantization_config=bnb_config,
        )
    else:
        # For full fine-tuning, we can speed up the model using Flash Attention
        # Only available on certain devices, see https://github.com/Dao-AILab/flash-attention#installation-and-features
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )

    # Prepare model for LoRA training if needed
    if USE_LORA or USE_QLORA:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

    # Load datasets
    dataset_path = args.dataset
    train_dataset = LlavaNextDataset(dataset_path, split="train")

    val_dataset = LlavaNextDataset(dataset_path, split="validation")

    train_dataset = train_dataset.dataset.select(range(10))
    val_dataset = val_dataset.dataset.select(range(20))

    # Verify the length of the small dataset
    print(len(train_dataset)) 

    # Initialize Wandb Logger
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

    # Initialize Early Stopping Callback
    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
    #config
    config = {
        "max_epochs": args.max_epochs,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_nodes": args.num_nodes,
        "warmup_steps": args.warmup_steps,
        "result_path": args.result_path,
        "verbose": args.verbose,
    }

    # Initialize the model module
    model_module = LlavaModelPLModule(
        config=config,
        processor=processor,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_collate_fn=train_collate_fn,
        eval_collate_fn=eval_collate_fn
    )

    args.gpus = list(map(int, args.gpus.split(',')))
    # Set default strategy if using multiple GPUs
    if len(args.gpus) > 1 and args.strategy is None:
        args.strategy = "ddp"  # Use DDP for multi-GPU training

    # Initialize Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        strategy=args.strategy,
        devices=args.gpus,
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        # callbacks=[PushToHubCallback(), early_stop_callback],
    )

    # Start training
    trainer.fit(model_module)