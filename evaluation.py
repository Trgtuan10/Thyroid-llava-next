import json
import os
from collections import Counter
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from datasets import load_from_disk, load_dataset
from dataset import LlavaNextDataset
import torch
from PIL import Image
import requests
import re


def extract_values(text):
    # Define regex patterns to extract the four sections
    patterns = {
        "TIRADS": r"TIRADS:\s*(.*?)\s*(?=FNAC|$)",
        "FNAC": r"FNAC:\s*(.*?)\s*(?=Histopathology|$)",
        "Histopathology": r"Histopathology:\s*(.*?)\s*(?=Result|$)",
        "Result": r"Result:\s*(.*)"
    }
    
    extracted_data = {}
    
    # Extract each part using the defined patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_data[key] = match.group(1).strip()
        else:
            extracted_data[key] = "Not Found"
    
    return extracted_data

def compare_results(extracted_model_answer, extracted_answer):
    comparison = []
    # Iterate over the four parts and compare values
    for key in ["TIRADS", "FNAC", "Histopathology", "Result"]:
        if extracted_model_answer.get(key) == extracted_answer.get(key):
            comparison.append(1)  # Same -> 1
        else:
            comparison.append(0)  # Different -> 0
    return comparison


def load_model():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")
    return processor, model

def run_tests():
    processor, model = load_model()
    
    dataset_path = "llava_medical_short_dataset"
    dataset = LlavaNextDataset(dataset_path, split="test")
    
    # Initialize counters
    total_comparisons = [0, 0, 0, 0]  # [TIRADS, FNAC, Histopathology, Result]
    correct_comparisons = [0, 0, 0, 0]

    for example in dataset:
        image = example["image"]
        answer = example["answer"]

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Predict the TIRADS classification, FNAC result, and potential diagnosis based on this thyroid ultrasound image with: TIRADS is a system that classifies thyroid nodules based on ultrasound features to assess malignancy risk, ranging from benign (TIRADS 1) to highly suspicious (TIRADS 5). FNAC is a procedure that uses a needle to collect cells from nodules for diagnosis, determining if they are benign or malignant. Histopathology examines tissue under a microscope to confirm malignancy, and malignancy refers to the presence of cancerous cells in a nodule."},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # Autoregressively complete the prompt
        output = model.generate(**inputs, max_new_tokens=256)

        model_answer = processor.decode(output[0], skip_special_tokens=True)

        # Extract values from both model_answer and answer
        extracted_model_answer = extract_values(model_answer)
        extracted_answer = extract_values(answer)

        # Compare the results
        comparison = compare_results(extracted_model_answer, extracted_answer)

        # Update counters
        for i in range(4):
            total_comparisons[i] += 1
            if comparison[i] == 1:
                correct_comparisons[i] += 1

    # Calculate the percentage of correct answers for each part
    accuracy_percentage = [
        (correct_comparisons[i] / total_comparisons[i]) * 100 if total_comparisons[i] > 0 else 0
        for i in range(4)
    ]

    # Output the results
    part_names = ["TIRADS", "FNAC", "Histopathology", "Result"]
    for i in range(4):
        print(f"Accuracy for {part_names[i]}: {accuracy_percentage[i]:.2f}%")

if __name__ == "__main__":
    run_tests()

        

