import os
import json

from transformers import BertTokenizerFast
import numpy as np

class DataGenerator:
    def __init__(self, path_tokenizer, list_data) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(path_tokenizer)

        self.list_origin_data = list_data
        self.list_encoded_data = [self.__get_instruct_contents(i) for i in list_data]
        self.list_instruction = [self.__get_instruction(i) for i in list_data]
        self.list_content = [self.__get_content(i) for i in list_data]
        self.max_len = len(list_data)

    def get_concated_data(self) -> np.ndarray:
        return np.concatenate(self.list_encoded_data)

    def __get_instruct_contents(self, data) -> list:
        result = "<type>" + data["essay_type"] + \
                "<sub>" + data["essay_main_subject"] + \
                "<gen>" + data["essay_sub_subject"]
        
        return self.tokenizer.encode(result)
    
    def __get_instruction(self, data) -> list:
        result = "<type>" + data["essay_type"] + \
                "<sub>" + data["essay_main_subject"]
        
        return self.tokenizer.encode(result, padding='max_length', max_length=512, truncation=True)[1:-1][::-1]
    
    def __get_content(self, data) -> list:
        result = data["essay_sub_subject"]

        return self.tokenizer.encode(result, padding='max_length', max_length=512, truncation=True)[1:-1][::-1]
    
    def __getitem__(self, index):
        return np.array(self.list_encoded_data[index])
    
    def __len__(self):
        return self.max_len
    
def load_dataset(path_dataset) -> DataGenerator:
    list_data = []

    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    list_data.append(data)
    
    data_generator = DataGenerator("klue/bert-base", list_data)

    return data_generator