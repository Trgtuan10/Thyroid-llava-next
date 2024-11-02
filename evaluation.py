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

def extract_TIRADS(text):
    tirads_pattern = r"TIRADS:\s*(\d)"
    match = re.search(tirads_pattern, text, re.IGNORECASE)
    if match:
        tirads_value = match.group(1).strip()
    else:
        tirads_value = "Not Found"
    return tirads_value

def extract_size(text):
    pattern = r"Size:\s*(\d+)\s*x\s*(\d+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        length = int(match.group(1))
        width = int(match.group(2))
        size = [length, width]
        return size
    else:
        return None

def extract_bbox(text):
    pattern = r"Bbox:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        w = int(match.group(3))
        h = int(match.group(4))
        x1 = x
        y1 = y
        x2 = x + w - 1
        y2 = y + h - 1
        extracted_data = {"Bbox": [x1, y1, x2, y2]}
    else:
        extracted_data = {"Bbox": "Not Found"}
    return extracted_data

def extract_results(text):
    pattern = r"Results:.*\((\d)\)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        result_category = int(match.group(1))
        return result_category
    else:
        return "Not Found"

def extract_postion(text):
    pattern = r"Position:\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        position = match.group(1).strip()
        return position
    else:
        return "Not Found"

def extract_all(text):
    patterns = {
        "TIRADS": r"TIRADS:\s*(\d+)",
        "Size": r"Size:\s*(\d+)\s*x\s*(\d+)",
        "Bbox": r"Bbox:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]",
        "Results": r"Results:.*\((\d)\)",
        "Position": r"Position:\s*(.*)"
    }

    extracted_data = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if key == "Size":
                extracted_data[key] = [int(match.group(1)), int(match.group(2))]
            elif key == "Bbox":
                x = int(match.group(1))
                y = int(match.group(2))
                w = int(match.group(3))
                h = int(match.group(4))
                x1 = x
                y1 = y
                x2 = x + w - 1
                y2 = y + h - 1
                extracted_data[key] = [x1, y1, x2, y2]
            else:
                extracted_data[key] = match.group(1).strip()
        else:
            extracted_data[key] = "Not Found"
    return extracted_data

def calculate_iou_size(pred_size, true_size):
    pred_length, pred_width = pred_size
    true_length, true_width = true_size

    pred_area = pred_length * pred_width
    true_area = true_length * true_width

    inter_length = min(pred_length, true_length)
    inter_width = min(pred_width, true_width)
    inter_area = inter_length * inter_width

    union_area = pred_area + true_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_iou_bbox(box1_coords, box2_coords):
    x1_box1, y1_box1, x2_box1, y2_box1 = box1_coords
    x1_box2, y1_box2, x2_box2, y2_box2 = box2_coords

    x1_inter = max(x1_box1, x1_box2)
    y1_inter = max(y1_box1, y1_box2)
    x2_inter = min(x2_box1, x2_box2)
    y2_inter = min(y2_box1, y2_box2)

    inter_width = max(0, x2_inter - x1_inter + 1)
    inter_height = max(0, y2_inter - y1_inter + 1)
    inter_area = inter_width * inter_height

    box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / float(union_area) if union_area > 0 else 0.0
    return iou

def load_model():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("TrgTuan10/Thyroid-llava-next-multi-prompt", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda")
    return processor, model

def run_tests():
    processor, model = load_model()
    dataset_path = "/workspace/lab/llava_medical_multi_question_dataset"
    dataset = LlavaNextDataset(dataset_path, split="test")
    #get 20 value from dataset
    dataset = [dataset[i] for i in range(min(20, len(dataset)))]


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

    for example in dataset:
        print(example)
        question_id = int(example.get("question_id"))
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
            model_position = extract_postion(model_answer)
            true_position = extract_postion(answer)
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
                iou_bbox = calculate_iou_bbox(model_bbox['Bbox'], true_bbox['Bbox'])
                scores['iou_bbox'] += iou_bbox
                sample_counts['bbox'] += 1
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
                iou_bbox = calculate_iou_bbox(model_data["Bbox"], true_data["Bbox"])
                scores['iou_bbox'] += iou_bbox
                sample_counts['bbox'] += 1
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
    avg_iou_bbox = scores['iou_bbox'] / sample_counts['bbox'] if sample_counts['bbox'] > 0 else 0
    avg_acc_position = scores['acc_position'] / sample_counts['position'] if sample_counts['position'] > 0 else 0

    return avg_acc_tirads, avg_iou_size, avg_iou_bbox, avg_acc_results, avg_acc_position, not_found_counts

if __name__ == "__main__":
    results = run_tests()
    print(f"Average Accuracy for TIRADS: {results[0]}")
    print(f"Average IoU for Size: {results[1]}")
    print(f"Average IoU for Bbox: {results[2]}")
    print(f"Average Accuracy for Results: {results[3]}")
    print(f"Average Accuracy for Position: {results[4]}")
    print(f"Counts of missing values: {results[5]}")

    