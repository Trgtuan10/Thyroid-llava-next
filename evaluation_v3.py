import json
import re
import numpy as np

def extract_TIRADS(text):
    tirads_pattern = r"TIRADS:\s*(\d)"
    match = re.search(tirads_pattern, text, re.IGNORECASE)
    if match:
        tirads_value = match.group(1).strip()
    else:
        tirads_value = "Not Found"
    return tirads_value

def extract_size(text):
    pattern =  r"Size:\s*([\d\.]+)\s*x\s*([\d\.]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        length = float(match.group(1))
        width = float(match.group(2))
        size = [length, width]
        return size
    else:
        return None
    
def extract_position(text):
    pattern = r"Position:\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        position = match.group(1).strip()
        return position
    else:
        return "Not Found"

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

def extract_class(text):
    pattern = r"Class:\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        class_value = int(match.group(1))
        return class_value
    else:
        return "Not Found"
    
def extract_conclusion(text):
    pattern = r"Conclusion:\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        conclusion = match.group(1).strip()
        return conclusion
    else:
        return "Not Found"

    
def extract_all(text):
    patterns = {
        "TIRADS": r"TIRADS:\s*(\d+)",
        "Size": r"Size:\s*([\d\.]+)\s*x\s*([\d\.]+)",
        "Bbox": r"Bbox:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]",
        "Conclusion": r"Conclusion:\s*(.*)",
        "Position": r"Position:\s*(.*)",
        "Class": r"Class:\s*(.*)"
    }

    extracted_data = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if key == "Size":
                extracted_data[key] = [float(match.group(1)), float(match.group(2))]
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

def calculate_map(predictions_bbox, ground_truth_bbox, iou_threshold=0.5):
    """Tính mAP với ngưỡng IoU cụ thể."""
    aps = []

    for cls_id in predictions_bbox.keys():
        pred_boxes = predictions_bbox.get(cls_id, [])
        gt_boxes = ground_truth_bbox.get(cls_id, [])
        
        # Chuyển predictions thành danh sách bbox
        pred_boxes = [(bbox, score) for bbox, score in pred_boxes]
        gt_boxes = [tuple(gt) for gt in gt_boxes]
        
        # Sắp xếp predictions theo độ tin cậy (confidence score)
        pred_boxes.sort(key=lambda x: x[1], reverse=True)
        
        tp = []  # True Positives
        fp = []  # False Positives
        gt_used = set()
        
        for pred_box, _ in pred_boxes:
            matched = False
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in gt_used:
                    continue
                
                iou = calculate_iou_bbox(pred_box, gt_box)
                if iou >= iou_threshold:
                    tp.append(1)
                    fp.append(0)
                    gt_used.add(gt_idx)
                    matched = True
                    break
            
            if not matched:
                tp.append(0)
                fp.append(1)
        
        # Cumulative sum for TP and FP
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        precisions = tp / (tp + fp + 1e-10)
        recalls = tp / (len(gt_boxes) + 1e-10)

        # Compute AP (Average Precision)
        ap = 0
        for i in range(len(precisions)):
            ap += precisions[i] * (recalls[i] - recalls[i - 1] if i > 0 else recalls[i])
        aps.append(ap)

    # Tính mean Average Precision
    return np.mean(aps)

def run_tests():
    # #load json 
    # with open("testcase/answer.json", "r") as f:
    #     data = f.readlines()
    # data = [json.loads(d) for d in data]

    data = []
    with open("testcase/answer.json", "r") as f:
        data = json.load(f)

    scores = {
        'acc_tirads': 0,
        'acc_conclusion': 0,
        'acc_size': 0,
        'iou_bbox': 0,
        'acc_position': 0
    }

    sample_counts = {
        'tirads': 0,
        'conclusion': 0,
        'size': 0,
        'bbox': 0,
        'position': 0
    }

    not_found_counts = {
        'tirads': 0,
        'conclusion': 0,
        'size': 0,
        'bbox': 0,
        'position': 0
    }

    predictions_bbox = {0: [], 1: []}
    ground_truth_bbox = {0: [], 1: []} 

    for example in data:
        question_id = int(example.get("question_id"))
        answer = example.get("answer")
        model_answer = example.get("model_answer")

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
                if model_size == true_size: 
                    scores['acc_size'] += 1
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
            #Class
            model_class = extract_class(model_answer)
            true_class = extract_class(answer)
            if model_bbox['Bbox'] != "Not Found" and true_bbox['Bbox'] != "Not Found":
                # iou_bbox = calculate_iou_bbox(model_bbox['Bbox'], true_bbox['Bbox'])
                # scores['iou_bbox'] += iou_bbox
                # sample_counts['bbox'] += 1
                confidences = 1.0
                #model_class in 0 or 1
                if model_class in [0, 1]:
                    predictions_bbox[model_class].append((model_bbox['Bbox'], confidences))
                if true_class in [0, 1]:
                    ground_truth_bbox[true_class].append(true_bbox['Bbox'])
                
            else:
                not_found_counts['bbox'] += 1

        elif question_id == 3:
            # Conclusion
            model_conclusion = extract_conclusion(model_answer)
            true_conclusion = extract_conclusion(answer)
            if model_conclusion != "Not Found" and true_conclusion != "Not Found":
                if model_conclusion == true_conclusion:
                    scores['acc_conclusion'] += 1
                sample_counts['conclusion'] += 1
            else:
                not_found_counts['conclusion'] += 1

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

            # Conclusion
            if model_data["Conclusion"] != "Not Found" and true_data["Conclusion"] != "Not Found":
                if model_data["Conclusion"] == true_data["Conclusion"]:
                    scores['acc_conclusion'] += 1
                sample_counts['conclusion'] += 1
            else:
                not_found_counts['conclusion'] += 1

            # Size
            if model_data["Size"] != "Not Found" and true_data["Size"] != "Not Found":
                if model_data["Size"] == true_data["Size"]:
                    scores['acc_size'] += 1
                sample_counts['size'] += 1
            else:
                not_found_counts['size'] += 1

            # Bbox
            if model_data["Bbox"] != "Not Found" and true_data["Bbox"] != "Not Found":
                # iou_bbox = calculate_iou_bbox(model_data["Bbox"], true_data["Bbox"])
                # scores['iou_bbox'] += iou_bbox
                # sample_counts['bbox'] += 1
                confidences = 1.0
                #model_class in 0 or 1
                model_class = int(model_data["Class"])
                true_class = int(true_data["Class"])

                if model_class in [0, 1]:
                    predictions_bbox[model_class].append((model_data["Bbox"], confidences))
                if true_class in [0, 1]:
                    ground_truth_bbox[true_class].append(true_data["Bbox"])

                
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
    avg_acc_conclusion = scores['acc_conclusion'] / sample_counts['conclusion'] if sample_counts['conclusion'] > 0 else 0
    avg_acc_size = scores['acc_size'] / sample_counts['size'] if sample_counts['size'] > 0 else 0
    #avg_iou_bbox = scores['iou_bbox'] / sample_counts['bbox'] if sample_counts['bbox'] > 0 else 0
    #Compute mAP-50, mAP-75, mAP-95
    ap_50 = calculate_map(predictions_bbox, ground_truth_bbox, 0.5)
    ap_75 = calculate_map(predictions_bbox, ground_truth_bbox, 0.75)
    ap_95 = calculate_map(predictions_bbox, ground_truth_bbox, 0.95)
    ap = [ap_50, ap_75, ap_95]
    avg_acc_position = scores['acc_position'] / sample_counts['position'] if sample_counts['position'] > 0 else 0

    return avg_acc_tirads, avg_acc_size, ap, avg_acc_conclusion, avg_acc_position, not_found_counts

if __name__ == "__main__":
    results = run_tests()
    print(f"Average Accuracy for TIRADS: {results[0]}")
    print(f"Average Accuracy for Size: {results[1]}")
    print("Average AP for IoU:")
    print(f"mAP-50: {results[2][0]}")
    print(f"mAP-75: {results[2][1]}")
    print(f"mAP-95: {results[2][2]}")
    print(f"Average Accuracy for Conclusion: {results[3]}")
    print(f"Average Accuracy for Position: {results[4]}")
    print(f"Counts of missing values: {results[5]}")
