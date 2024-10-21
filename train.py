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

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        # Unpack the batch
        input_ids, attention_mask, pixel_values, image_sizes, answers = batch

        # Generate predictions using autoregression
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_new_tokens=self.config.get("max_new_tokens", 128)
        )

        # Decode the generated tokens into text
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            # No regex is needed here, directly compare
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        # Log validation edit distance
        self.log("val_edit_distance", np.mean(scores), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return scores

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
                    {"type": "text", "text": "Predict the TIRADS classification, FNAC result, and potential diagnosis based on this thyroid ultrasound image with: TIRADS is a system that classifies thyroid nodules based on ultrasound features to assess malignancy risk, ranging from benign (TIRADS 1) to highly suspicious (TIRADS 5). FNAC is a procedure that uses a needle to collect cells from nodules for diagnosis, determining if they are benign or malignant. Histopathology examines tissue under a microscope to confirm malignancy, and malignancy refers to the presence of cancerous cells in a nodule. "},
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

    # Loop through the examples in the batch
    for example in examples:
        image = example["image"]          # Extract the image
        answer = example["answer"]        # Extract the assistant's answer (text)

        # Add image to the batch
        images.append(image)

        # Create conversation structure (without the assistant's answer for evaluation)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Reference to the image
                    {"type": "text", "text": "Predict the TIRADS classification, FNAC result, and potential diagnosis based on this thyroid ultrasound image with: TIRADS is a system that classifies thyroid nodules based on ultrasound features to assess malignancy risk, ranging from benign (TIRADS 1) to highly suspicious (TIRADS 5). FNAC is a procedure that uses a needle to collect cells from nodules for diagnosis, determining if they are benign or malignant. Histopathology examines tissue under a microscope to confirm malignancy, and malignancy refers to the presence of cancerous cells in a nodule."},  # User's question
                ],
            }
        ]

        # Apply the conversation template with generation prompt enabled for evaluation
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        texts.append(text_prompt)

        # Save the answer to compare later
        answers.append(answer)

    # Process text and images together
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Return the necessary tensors for the model's input, along with ground truth answers for evaluation
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_sizes"], answers

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
    parser.add_argument("--strategy", type=str, default=None)
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
    REPO_ID = "TrgTuan10/Thyroid-llava-next"
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
        callbacks=[PushToHubCallback(), early_stop_callback],
    )

    # Start training
    trainer.fit(model_module)
