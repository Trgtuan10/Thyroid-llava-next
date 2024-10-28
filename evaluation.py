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

    # Search for the TIRADS value in the text
    match = re.search(tirads_pattern, text, re.IGNORECASE)
    
    # If a match is found, return the TIRADS value, otherwise return 'Not Found'
    if match:
        tirads_value = match.group(1).strip()  # Extract the first matched group (TIRADS number)
    else:
        tirads_value = "Not Found"
    
    return tirads_value

def extract_position(text):
    """
    Extract the size of the nodule in millimeters (length and width) from the text and return them as a list of two integers.
    
    Args:
        text (str): The input text containing the position information.
        id (int): The ID associated with the example.
        
    Returns:
        position (list): A list containing the length and width as integers [length, width].
        id (int): The corresponding ID.
    """
    # Define the pattern to extract nodule size in the format (length x width)
    pattern = r"Position:\s*(\d+)\s*x\s*(\d+)\s*mm"
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        length = int(match.group(1))  # Extract and convert length to integer
        width = int(match.group(2))   # Extract and convert width to integer
        position = [length, width]    # Store length and width in a list
        return position
    else:
        return None

def extract_bbox(text):
    """
    Extract the bounding box coordinates from the text.
    
    Args:
        text (str): The input text containing the bounding box information.
        id (int): The ID associated with the example.
        
    Returns:
        extracted_data (dict): Dictionary containing the bounding box as a list of tuples (x1, y1, x2, y2).
        id (int): The corresponding ID.
    """
    # Define the pattern to extract bounding box coordinates in the format (x1, y1, x2, y2)
    pattern = r"Bbox:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        x1 = int(match.group(1))
        y1 = int(match.group(2))
        x2 = int(match.group(3))
        y2 = int(match.group(4))
        extracted_data = {"Bbox": [(x1, y1), (x2, y2)]}
    else:
        extracted_data = {"Bbox": "Not Found"}
    
    return extracted_data

def extract_results(text):
    """
    Extract the final diagnostic conclusion based on the doctor's result from the text.
    
    Args:
        text (str): The input text containing the final result.
        id (int): The ID associated with the example.
        
    Returns:
        extracted_data (dict): Dictionary containing the result category in format {category (number)}.
        id (int): The corresponding ID.
    """
    # Define the pattern to extract the final result in format {category (number)}
    pattern = r"Results:\s*(\d)\s*"
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        result_category = int(match.group(1))
        extracted_data = {"Results": result_category}
    else:
        extracted_data = {"Results": "Not Found"}
    
    return extracted_data

def extract_all(text):
    """
    Extract TIRADS, Position, Bbox, and Result from the input text.
    
    Args:
        text (str): The input text.
    
    Returns:
        dict: Extracted data for TIRADS, Position, Bbox, and Results.
    """
    # Define regex patterns to extract the relevant sections
    patterns = {
        "TIRADS": r"TIRADS:\s*(\d+)",
        "Position": r"Position:\s*(\d+)\s*x\s*(\d+)\s*mm",
        "Bbox": r"Bbox:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
        "Results": r"Results:\s*(\d)"
    }
    
    extracted_data = {}
    
    # Extract each part using the defined patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if key == "Position":
                # Return position as a list [length, width]
                extracted_data[key] = [int(match.group(1)), int(match.group(2))]
            elif key == "Bbox":
                # Return bbox as a list of tuples [(x1, y1), (x2, y2)]
                extracted_data[key] = [(int(match.group(1)), int(match.group(2))),
                                       (int(match.group(3)), int(match.group(4)))]
            else:
                extracted_data[key] = match.group(1).strip()  # For TIRADS and Results
        else:
            extracted_data[key] = "Not Found"
    
    return extracted_data


def calculate_iou_position(pred_position, true_position):
    """
    Calculate IoU for position sizes (length x width).
    
    Args:
        pred_position (list): Predicted position as [length, width].
        true_position (list): Ground truth position as [length, width].
    
    Returns:
        float: IoU score.
    """
    # Unpack predicted and true position lengths and widths
    pred_length, pred_width = pred_position
    true_length, true_width = true_position
    
    # Calculate the area of the predicted and true rectangles
    pred_area = pred_length * pred_width
    true_area = true_length * true_width
    
    # Calculate the overlap (intersection) area in each dimension
    inter_length = min(pred_length, true_length)
    inter_width = min(pred_width, true_width)
    
    # Calculate the area of the intersection
    inter_area = inter_length * inter_width
    
    # Calculate the area of the union
    union_area = pred_area + true_area - inter_area
    
    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_iou_bbox(box1, box2):
    """
    Calculate IoU for bounding boxes.

    Args:
        box1 (dict): Dictionary containing the bounding box as a list of tuples [x1, y1, x2, y2].
        box2 (dict): Dictionary containing the bounding box as a list of tuples [x1, y1, x2, y2].

    Returns:
        float: IoU score.
    """
    # Check if both bounding boxes are valid
    if box1.get("Bbox") == "Not Found" or box2.get("Bbox") == "Not Found":
        return 0.0  # If either box is not found, IoU is 0

    # Extract coordinates from the bounding box dictionaries
    x1_box1, y1_box1, x2_box1, y2_box1 = box1["Bbox"]
    x1_box2, y1_box2, x2_box2, y2_box2 = box2["Bbox"]

    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(x1_box1, x1_box2)
    y1_inter = max(y1_box1, y1_box2)
    x2_inter = min(x2_box1, x2_box2)
    y2_inter = min(y2_box1, y2_box2)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, x2_inter - x1_inter + 1)
    inter_height = max(0, y2_inter - y1_inter + 1)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

    # Calculate the area of the union of the two boxes
    union_area = box1_area + box2_area - inter_area

    # Compute IoU (Intersection over Union)
    iou = inter_area / float(union_area) if union_area > 0 else 0.0
    return iou




def load_model():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")
    return processor, model

def run_tests():
    processor, model = load_model()
    
    dataset_path = "llava_medical_short_dataset"
    dataset = LlavaNextDataset(dataset_path, split="test")
    
    # define value of each part with TIRADS abd results will count by f1 score, postion and bbox will count by IoU
    # define a list to store the results
    # Extract values from both model_answer and answer
    scores = {
        'f1_tirads': 0,
        'f1_results': 0,
        'iou_position': 0,
        'iou_bbox': 0
    }
    
    sample_counts = {
        'tirads': 0,
        'results': 0,
        'position': 0,
        'bbox': 0
    }
    

    for example in dataset:
        question_id = example["question_id"]
        question = example["question"]
        image = example["image"]
        answer = example["answer"]

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
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

        # Autoregressively complete the prompt
        output = model.generate(**inputs, max_new_tokens=256)

        model_answer = processor.decode(output[0], skip_special_tokens=True)

        # Extract values from both model_answer and answer
        if question_id == 0:
            # Extract and compare TIRADS
            model_tirads= extract_TIRADS(model_answer)
            true_tirads = extract_TIRADS(answer)
            
            # Increment F1 score for TIRADS if prediction matches
            if model_tirads == true_tirads:
                scores['f1_tirads'] += 1
            sample_counts['tirads'] += 1

        if question_id == 1:
            model_position = extract_position(model_answer)
            true_position = extract_position(answer)
            
            if model_position and true_position:
                iou_position = calculate_iou_position(model_position, true_position)  # Adjust your IoU calculation here
                scores['iou_position'] += iou_position
            sample_counts['position'] += 1

        if question_id == 2:
            model_bbox = extract_bbox(model_answer)
            true_bbox = extract_bbox(answer)
            
            if model_bbox and true_bbox:
                iou_bbox = calculate_iou_bbox(model_bbox['Bbox'], true_bbox['Bbox'])  # Adjust IoU calculation
                scores['iou_bbox'] += iou_bbox
            sample_counts['bbox'] += 1

        if question_id == 3:
            # Extract and compare Results
            model_results, _ = extract_results(model_answer)
            true_results, _ = extract_results(answer)
            
            # Increment F1 score for Results if prediction matches
            if model_results == true_results:
                scores['f1_results'] += 1
            sample_counts['results'] += 1

        if question_id == 4:
            # Extract and compare all parts
            model_data = extract_all(model_answer)
            true_data = extract_all(answer)

            # Loop through the keys in model_data (TIRADS, Position, Bbox, Results)
            for key in model_data:
                if key == "TIRADS" or key == "Results":
                    # Compare TIRADS and Results directly
                    if model_data[key] == true_data[key]:
                        scores[f"f1_{key.lower()}"] += 1
                    sample_counts[key.lower()] += 1
                elif key == "Position":
                    # Compare positions using IoU if both are found
                    model_position = model_data[key]
                    true_position = true_data[key]
                    if model_position != "Not Found" and true_position != "Not Found":
                        iou_position = calculate_iou_position(model_position, true_position)
                        scores['iou_position'] += iou_position
                    sample_counts['position'] += 1
                elif key == "Bbox":
                    # Compare bounding boxes using IoU if both are found
                    model_bbox = model_data[key]
                    true_bbox = true_data[key]
                    if model_bbox != "Not Found" and true_bbox != "Not Found":
                        iou_bbox = calculate_iou_bbox({"Bbox": model_bbox}, {"Bbox": true_bbox})
                        scores['iou_bbox'] += iou_bbox
                    sample_counts['bbox'] += 1


    # Calculate average F1 scores and IoU scores
    avg_f1_tirads = scores['f1_tirads'] / sample_counts['tirads'] if sample_counts['tirads'] > 0 else 0
    avg_f1_results = scores['f1_results'] / sample_counts['results'] if sample_counts['results'] > 0 else 0
    avg_iou_position = scores['iou_position'] / sample_counts['position'] if sample_counts['position'] > 0 else 0
    avg_iou_bbox = scores['iou_bbox'] / sample_counts['bbox'] if sample_counts['bbox'] > 0 else 0

    return avg_f1_tirads, avg_iou_position, avg_iou_bbox, avg_f1_results

        

if __name__ == "__main__":
    results = run_tests()
    print(f"Average F1 Score for TIRADS: {results[0]}")
    print(f"Average IoU for Position: {results[1]}")
    print(f"Average IoU for Bbox: {results[2]}")
    print(f"Average F1 Score for Results: {results[3]}")
    

        

