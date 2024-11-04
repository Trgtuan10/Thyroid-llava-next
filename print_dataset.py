import json
from collections import Counter

# Load data from a JSON file
def load_data(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        return None

# Function to extract the values
def extract_values(data):
    extracted_data = []
    count = 0
    for item in data:
        count += 1
        tirads = results = position = bbox = size = None

        # Check each conversation in the item
        for conversation in item.get("conversations", []):
            response_text = conversation["value"].strip()
            if count == 1:
                print(response_text)

            # Extract TIRADS
            if "TIRADS:" in response_text:
                tirads = response_text.split("TIRADS: ")[1].split()[0].strip()
            
            # Extract Results (FNAC result, usually as "Results: The FNAC (2)")
            if "Results:" in response_text:
                print("ok")
                try:
                    results = response_text.split("Results:")[1].split("\n")[0].strip()
                    print(results)
                except IndexError:
                    pass
            
            # Extract Position
            if "Position:" in response_text:
                position = response_text.split("Position: ")[1].split("\n")[0].strip()
            
            # Extract Bbox
            if "Bbox:" in response_text:
                try:
                    bbox = response_text.split("Bbox: ")[1].split("\n")[0].strip("[] ")
                except IndexError:
                    pass
            
            # Extract Size
            if "Size:" in response_text:
                size = response_text.split("Size: ")[1].split("\n")[0].strip()

        # Add extracted data even if some fields are missing
        extracted_data.append({
            "TIRADS": tirads,
            "Results": results,
            "Position": position,
            "Bbox": bbox,
            "Size": size
        })
    print(f"Extracted {count} items.")
    return extracted_data

# Function to count unique values in each part
def count_values(extracted_data):
    tirads_values = [item["TIRADS"] for item in extracted_data if item["TIRADS"]]
    results_values = [item["Results"] for item in extracted_data if item["Results"]]
    position_values = [item["Position"] for item in extracted_data if item["Position"]]
    bbox_values = [item["Bbox"] for item in extracted_data if item["Bbox"]]
    size_values = [item["Size"] for item in extracted_data if item["Size"]]
    
    # Count occurrences of each unique value
    tirads_count = Counter(tirads_values)
    results_count = Counter(results_values)
    position_count = Counter(position_values)
    bbox_count = Counter(bbox_values)
    size_count = Counter(size_values)
    
    return tirads_count, results_count, position_count, bbox_count, size_count

# Function to display counts and optionally save to a file
def display_and_save_counts(tirads_count, results_count, position_count, bbox_count, size_count, output_file=None):
    print("TIRADS counts:", tirads_count)
    print("Results counts:", results_count)
    print("Position counts:", position_count)
    print("Bbox counts:", bbox_count)
    print("Size counts:", size_count)
    
    if output_file:
        counts = {
            "TIRADS": dict(tirads_count),
            "Results": dict(results_count),
            "Position": dict(position_count),
            "Bbox": dict(bbox_count),
            "Size": dict(size_count)
        }
        with open(output_file, "w") as file:
            json.dump(counts, file, indent=4)
        print(f"Counts saved to {output_file}")

# Main execution
file_path = "llava_medical_final_longer.json"  # Path to the input JSON file
output_file = "counts_output.json"      # Optional: Path to save the counts

data = load_data(file_path)
if data:
    extracted_data = extract_values(data)
    tirads_count, results_count, position_count, bbox_count, size_count = count_values(extracted_data)
    display_and_save_counts(tirads_count, results_count, position_count, bbox_count, size_count, output_file)

response_text = "The image is analyzed as follows:\nResults: The FNAC (2)"
results = response_text.split("Results:")[1].split("\n")[0].strip()
print(results)
