from datasets import Dataset, DatasetDict
import json
from PIL import Image
import argparse
# Process the data to include 'question', 'image', and 'answer'
def process_data(data, data_dir="images"):
    processed_data = {
        "question_id": [],
        "question": [],
        "image": [],
        "answer": []
    }
    for entry in data:
        image = entry["images"]
        answer = entry["answer"]
        question = entry["question"] 
        question_id = entry["question_id"]  # Assuming the last conversation is the question
        processed_data["image"].append(Image.open(f"{data_dir}/{image}"))
        processed_data["answer"].append(answer)
        processed_data["question"].append(question)
        processed_data["question_id"].append(question_id)
    return processed_data

def create_dataset(train_json, test_json, train_dir, test_dir, save_path="dataset"):
    # Load JSON files
    with open(train_json, "r") as f:
        train_json_data = json.load(f)
    with open(test_json, "r") as f:
        test_json_data = json.load(f)

    # Process train and test data
    train_data = process_data(train_json_data, train_dir)
    test_data = process_data(test_json_data, test_dir)

    # Create DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_data)
    })

    # Save DatasetDict
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/workspace/lab/thyroidv2/thyroid_dataset_v1.0.0/train/images")
    parser.add_argument("--test_dir", type=str, default="/workspace/lab/thyroidv2/thyroid_dataset_v1.0.0/test/images")
    parser.add_argument("--output_dir", type=str, default="../llava_medical")
    parser.add_argument("--train_json", type=str, default="/workspace/lab/Thyroid-llava-next/datasetv2/llava_medical_v2_train.json")
    parser.add_argument("--test_json", type=str, default="/workspace/lab/Thyroid-llava-next/datasetv2/llava_medical_v2_test.json")
    args = parser.parse_args()

    create_dataset(args.train_json, args.test_json, args.train_dir, args.test_dir, args.output_dir)

