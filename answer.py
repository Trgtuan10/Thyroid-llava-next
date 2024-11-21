import json
import numpy as np
import os
from collections import Counter
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from datasets import load_from_disk, load_dataset
from dataset import LlavaNextDataset
import torch
from PIL import Image
import requests
import re


def load_model():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("TrgTuan10/Thyroid-llava-next-multi-prompt", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda")
    return processor, model

def clean_answer(text):
    # Remove everything up to and including [/INST] and any trailing unwanted data
    text = re.sub(r'.*\[/INST\]\s*', '', text, flags=re.DOTALL)
    # Extract only the primary content of the answer without repeated patterns
    text = re.split(r'<\\s>|<\\"|\\\\n|\\\\', text)[0].strip()
    return text

def run_tests():
    processor, model = load_model()
    dataset_path = "/workspace/lab/llava_medical_multi_question_dataset"
    dataset = LlavaNextDataset(dataset_path, split="test")

    # Set eos_token_id as pad_token_id if needed
    eos_token_id = processor.tokenizer.eos_token_id
    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else eos_token_id

    for example in dataset:
        question_id = example.get("question_id")
        question = example.get("question")
        image = example.get("image")
        answer = example.get("answer")
        # Prepare prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

        # Generate model answer
        output = model.generate(**inputs, max_new_tokens=256)
        model_answer = processor.decode(output[0], skip_special_tokens=True)

        # Clean up the answer
        model_answer = clean_answer(model_answer)

        #write to json file for checking
        with open("answer.json", "a") as f:
            f.write(json.dumps({"question_id": question_id, "answer": answer, "model_answer": model_answer}) + "\n")


if __name__ == "__main__":
    # run_tests()
    # text = "[INST]  \nFrom this image, provide the TIRADS level of the nodule (from 0  to 5), where higher levels indicate a higher risk of cancer. The output should be a number from 0 to 5. [/INST] The image is analyzed as follows:\nTIRADS: 5 0.0001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"
    # print(clean_answer(text))
    dataset_path = "/workspace/lab/llava_medical_multi_question_dataset"

    dataset = LlavaNextDataset(dataset_path, split="test")

    #count number data point
    print(len(dataset))