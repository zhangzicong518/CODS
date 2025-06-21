from datasets import load_from_disk
import json

data = load_from_disk("/home/zicong/CODS/evaluate/combined_dataset")

test_data = data.filter(lambda x: x['data_type'] == 'few_shot')

print("Dataset Overview:")
print(test_data[0])