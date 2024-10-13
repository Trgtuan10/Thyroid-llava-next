from datasets import load_from_disk, DatasetDict, Dataset
from torch.utils.data import Dataset
from typing import Any, Dict

def load_dataset(path: str, percent: float = 1) -> DatasetDict:
    def sample_percent_data(dataset, percent):
        sample_size = int(percent * len(dataset))
        sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
        return sampled_dataset
    
    dataset = load_from_disk(path)

    new_train_dataset = sample_percent_data(dataset['train'])
    new_validation_dataset = sample_percent_data(dataset['validation'])
    new_test_dataset = sample_percent_data(dataset['test'])

    # Create a new DatasetDict with 10% of the original size
    new_dataset_dict = DatasetDict({
        'train': new_train_dataset,
        'validation': new_validation_dataset,
        'test': new_test_dataset
    })

    return new_dataset_dict

class LlavaNextDataset(Dataset):
    """
    PyTorch Dataset for LLaVa-NeXT, modified for question, image, and answer fields.
    
    Each row consists of an image path (png/jpg/jpeg), a question (text), and an answer (text).
    """

    def __init__(self, dataset_name_or_path: str, split: str = "train"):
        super().__init__()

        self.split = split

        # Load dataset using HuggingFace's load_dataset
        self.dataset = load_from_disk(dataset_name_or_path)[self.split]
        self.dataset_length = len(self.dataset)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image: The image file path
            question: The question (input prompt)
            answer: The answer (ground truth text)
        """
        sample = self.dataset[idx]

        # Extract image, question, and answer
        image = sample["image"]          # Path to the image
        question = sample["question"]    # User's question
        answer = sample["answer"]        # Assistant's answer (ground truth)

        return {
            "image": image,
            "question": question,
            "answer": answer
        }
