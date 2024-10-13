from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from datasets import DatasetDict
import torch
import datetime

from dataset import load_dataset, LlavaNextDataset
from model import LlavaModelPLModule


MAX_LENGTH = 256
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
REPO_ID = "TrgTuan10/Thyroid-llava-next"
WANDB_PROJECT = "LLaVaNeXT-Thyroid"

now = datetime.datetime.now()
WANDB_NAME = "thyroid-" + now.strftime("%Y-%m-%d-%H-%M-%S")

USE_LORA = True
USE_QLORA = False

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
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )

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

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

train_dataset = LlavaNextDataset("llava_medical_dataset",  split="train")
val_dataset = LlavaNextDataset("llava_medical_dataset", split="validation")

def train_collate_fn(examples):
    images = []
    texts = []

    for example in examples:
        image = example["image"]
        question = example["question"]
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

    # Loop through the examples in the batch
    for example in examples:
        image = example["image"]          # Extract the image
        question = example["question"]    # Extract the user's question (text)
        answer = example["answer"]        # Extract the assistant's answer (text)

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

        # Save the answer to compare later
        answers.append(answer)

    # Process text and images together
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Return the necessary tensors for the model's input, along with ground truth answers for evaluation
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_sizes"], answers

config = {"max_epochs": 10,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
}

model_module = LlavaModelPLModule(config, processor, model)

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from huggingface_hub import HfApi

api = HfApi()

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

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
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

trainer.fit(model_module)