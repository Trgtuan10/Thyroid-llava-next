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
        answer = entry["conversations"][-1]["value"]  # Assuming the last conversation is the answer
        question = entry["conversations"][0]["value"]  # Assuming the last conversation is the question
        question_id = entry["question_id"]  # Assuming the last conversation is the question
        processed_data["image"].append(Image.open(f"{data_dir}/{image}"))
        processed_data["answer"].append(answer)
        processed_data["question"].append(question)
        processed_data["question_id"].append(question_id)
    return processed_data

if __name__ == "__main__":
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../images")
    parser.add_argument("--output_dir", type=str, default="../llava_medical_multi_question_dataset")
    args = parser.parse_args()

    with open("../llava_medical_final_longer.json", "r") as file:
        try:
            json_data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")

    train_size = int(0.8 * len(json_data))
    val_size = int(0.1 * len(json_data))

    train_data = json_data[:train_size]
    validation_data = json_data[train_size:train_size + val_size]
    test_data = json_data[train_size + val_size:]

    # Create Dataset objects from processed data
    train_dataset = Dataset.from_dict(process_data(train_data, data_dir=args.data_dir))
    validation_dataset = Dataset.from_dict(process_data(validation_data, data_dir=args.data_dir))
    test_dataset = Dataset.from_dict(process_data(test_data, data_dir=args.data_dir))

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    # Now you can use the dataset_dict for training LLaVA
    print(dataset_dict)

    # Save the dataset_dict to disk
    dataset_dict.save_to_disk(args.output_dir)