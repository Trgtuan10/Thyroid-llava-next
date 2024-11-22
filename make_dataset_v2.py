from datasets import load_dataset
import json

def make_full_conversation(image, info, annotations):
    TIRADS_question = "From this image, provide the TIRADS level of the nodule (from 0  to 5), where higher levels indicate a higher risk of cancer. The output should be a number from 0 to 5."
    PositionAndSizing_question = "From this image, provide the position and the actual size of the nodule in millimeters (length x width). The output of size should be in format (length x width)"
    BBox_question = "From this image, provide the position and type of the nodule in the thyroid gland. The Bbox should be in format (x,y,w,h)"
    Conclusion_question = "From this image, provide the final diagnostic conclusion from the doctor using the following categories: 1 = Cancer MBH, 2 = Cancer FNAC, 3 = Benign FNAC, 4 = Benign MBH."
    All_question = "From this thyroid ultrasound image, provide a prediction for the TIRADS classification (0-5) to assess the malignancy risk, the size of the nodule in millimeters (length x width), the bounding box coordinates around the nodule (x,y,w,h),  the final diagnosis category (1 = Cancer MBH, 2 = Cancer FNAC, 3 = Benign FNAC, 4 = Benign MBH) and characteristics of the nodule."

    #questions = [TIRADS_question, PositionAndSizing_question, BBox_question, Result_question, Characteristics_question, All_question]
    questions = [TIRADS_question, PositionAndSizing_question, BBox_question, Conclusion_question , All_question]
    answers = []

    for question in questions:
        gpt_value = "The image is analyzed as follows:"

        # Get patient information
        patient_id = str(image['patient_id'])
        
        #convert with 8 digits
        patient_id = patient_id.zfill(8)
        print("patient_id:", patient_id)

        # If patient_id is not in info, return None
        if patient_id not in info:
            print("Patient ID not found:", patient_id)
            return None
        value = info[patient_id]["nodule_1"]

        #get Bbox
        image_id = image['id']
        #find Bbox around image_id
        bbox = annotations[image_id-1]['bbox']
        cls = annotations[image_id-1]['category_id']
        if bbox is None:
            print("There is no Bbox")
            return None

 
        position = value["Position"]
        TIRADS = value["TIRADS"]
        conclusion = value["Conclusion"]
        width = value["Width"]
        height = value["Height"]
        size = f"{width} x {height}"

        if position is None or TIRADS is None or conclusion is None or width is None or height is None:
            print("Missing information for image:", image['id'])
            return None
    
        question_id = 0
        if question == TIRADS_question:
            gpt_value += f"\nTIRADS: {TIRADS}"
            question_id = 0
        if question == PositionAndSizing_question:
            gpt_value += f"\nPosition: {position}\nSize: {size}"
            question_id = 1
        if question == BBox_question:
            gpt_value += f"\nBbox: {bbox} Class: {cls}"
            question_id = 2
        if question == Conclusion_question:
            gpt_value += f"\nConclusion: {conclusion}"
            question_id = 3
      
        if question == All_question:
            #gpt_value += f"\nTIRADS: {TIRADS}\nPosition: {position}\nSize: {size}\nBbox: {bbox}\nResults: {result}\nPhonological: {phonological_info}, Nuclear morphology: {nuclear_morphology_info}, Nuclear border: {nuclear_border_info}, Calcification: {calcification_info}"
            gpt_value += f"\nTIRADS: {TIRADS}\nPosition: {position}\nSize: {size}\nBbox: {bbox} \nClass: {cls}\nConclusion: {conclusion}"
            question_id = 4
    
        # Create the JSON object for this image
        json_obj = {
            "id": image['id'],
            "question_id": question_id,
            "images": image['file_name'],
            "question": question,
            "answer": gpt_value
        }
        print("json_obj:", json_obj)

        # Append to the final list
        answers.append(json_obj)
    return answers
  
    

# Load the JSON dataset file
file_path = "/workspace/lab/thyroidv2/thyroid_dataset_v1.0.0/train/train_annotations.json"

# Mở và đọc file
with open(file_path, "r", encoding="utf-8") as file:
    dataset = json.load(file)
print("Dataset loaded:", dataset.keys())

# Get the patient info from the dataset
info = dataset['info']

# Get the 'images' column from the dataset
images = dataset["images"]

#get annotation column from the dataset
annotations = dataset["annotations"]

# Initialize a list to hold the final data structure
json_output = []

count = 0


with open("llava_medical_v2_train.json", "a", encoding="utf-8") as outfile:
    for image in images:
        print("Processing image:", image['id'])
        img_id = image['id']
        img_file = image['file_name']

        # Generate the conversation
        json_file = make_full_conversation(image, info, annotations)

        if json_file is None:
            print("Skipping image:", img_id)
            continue

        # Ghi từng json_obj vào file chính
        for json_obj in json_file:
            json_output.append(json_obj)
            count += 1
            # Ghi json_obj vào file "llava_medical_final_longer.json"
            json.dump(json_obj, outfile, ensure_ascii=False, indent=4)
            outfile.write("\n")  # Thêm dòng mới sau mỗi đối tượng JSON để dễ đọc


# # Save the list as a JSON file in the correct format
# with open('llava_medical_final_longer.json', 'w', encoding='utf-8') as f:
#     json.dump(json_output, f, ensure_ascii=False, indent=4)

print("Processed dataset saved as llava_medical_v2_train.json")
print("Total images processed:", count)
