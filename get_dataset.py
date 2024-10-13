from datasets import load_dataset
import json
from PIL import Image
from datasets import DatasetDict, Dataset

def get_answer(image, info):
    answer = "This is the prediction of the disease from this photo:"
    
    patient_id = str(image['patient_id'])
    if patient_id not in info:
        return None
    info_patient = info[patient_id]
    info_patient = {key: value for key, value in info_patient.items() if key.startswith('Nhan')}
    for key, value in info_patient.items():
        answer += f"\n{key}: {value}"

    slice_info = image.get('slice')
    position_info = image.get('position')
    parts_info = image.get('parts')
    phonological_info = image.get('phonological')
    nuclear_morphology_info = image.get('nuclear_morphology')
    nuclear_border_info = image.get('nuclear_border')
    calcification_info = image.get('calcification')

    if slice_info:
        answer += f"\nThe slice of the image is {slice_info}."
    if position_info:
        answer += f"\nPosition: {position_info}."
    if parts_info:
        answer += f"\nParts of the lesion: {parts_info}."
    if phonological_info:
        answer += f"\nPhonological structure: {phonological_info}."
    if nuclear_morphology_info:
        answer += f"\nNuclear morphology: {nuclear_morphology_info}."
    if nuclear_border_info:
        answer += f"\nNuclear border: {nuclear_border_info}."
    if calcification_info:
        answer += f"\nCalcification: {calcification_info}."
    
    return answer

def create_training_data(json_path: str):
    def process_data(data):
        processed_data = {
            "image": [],
            "answer": []
        }
        for entry in data:
            image = entry["images"]
            answer = entry["answer"]  # Assuming the last conversation is the answer
            processed_data["image"].append(Image.open(f"../images/{image}"))
            processed_data["answer"].append(answer)
        return processed_data
    with open(json_path, 'r') as f:
        json_data = json.load(f) 
    train_size = int(0.8 * len(json_data))
    val_size = int(0.1 * len(json_data))

    train_data = json_data[:train_size]
    validation_data = json_data[train_size:train_size + val_size]
    test_data = json_data[train_size + val_size:]

    # Create Dataset objects from processed data
    train_dataset = Dataset.from_dict(process_data(train_data))
    validation_dataset = Dataset.from_dict(process_data(validation_data))
    test_dataset = Dataset.from_dict(process_data(test_data))

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    print(dataset_dict)
    dataset_dict.save_to_disk('llava_medical_dataset')
    

if __name__ == "__main__":
    dataset = load_dataset('json', data_files='../v2.json')
    info = dataset['train'][0]['info']
    images = dataset['train'][0]["images"]

    json_output = []
    for image in images:
        img_id = image['id']
        img_file = image['file_name']
        answer = get_answer(image, info)
        if answer is None:
            continue

        json_obj = {
            "id": img_id,
            "images": img_file,
            "answer": answer
        }
        json_output.append(json_obj)

    with open('llava_medical.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)

    print("Processed dataset saved as llava_medical.json")
