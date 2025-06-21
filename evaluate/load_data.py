from datasets import load_dataset
import random
from collections import defaultdict
from datasets import Dataset, Features, Image, Value, ClassLabel

import json
data = json.load(open("few_shot_for_gradient.json", "r"))

# Load dataset
dataset = load_dataset("/home/zicong/DATASETS/MMBench", "en")
dev = dataset["dev"]

# Organize dev data by category
dev_by_category = defaultdict(list)
for item in dev:
    dev_by_category[item['category']].append(item)

# Sample 20 items for each category as few-shot, rest as test
few_shot_samples = []
test_samples = []
categories = list(dev_by_category.keys())
 
for category in categories:
    category_items = dev_by_category[category]
    # Shuffle the items for random sampling
    random.shuffle(category_items)
    
    # Take first 20 as few-shot (or all if less than 20)
    few_shot_count = min(20, len(category_items))
    few_shot_samples.extend(category_items[:few_shot_count])
    
    # Take the rest as test
    test_samples.extend(category_items[few_shot_count:])

import json

# Combine few-shot and test data with labels
combined_data = []

# Add few-shot data with labels
for item in few_shot_samples:
    item_copy = dict(item)
    item_copy['data_type'] = 'few_shot'
    combined_data.append(item_copy)

# Add test data with labels (now from remaining dev items)
for item in test_samples:
    item_copy = dict(item)
    item_copy['data_type'] = 'test'
    combined_data.append(item_copy)

# Sort by category, then by data_type (few_shot first, then test)
combined_data.sort(key=lambda x: (x['category'], x['data_type']))

# Create a Hugging Face dataset
combined_dataset = Dataset.from_list(combined_data)

# Save as dataset
output_path = "/home/zicong/CODS/evaluate/combined_dataset"
combined_dataset.save_to_disk(output_path)

print(f"Saved {len(combined_data)} items to {output_path}")
print(f"Total few-shot samples: {len(few_shot_samples)}")
print(f"Total test samples: {len(test_samples)}")
print("\nData structure:")
for category in sorted(set(item['category'] for item in combined_data)):
    few_shot_count = sum(1 for item in combined_data if item['category'] == category and item['data_type'] == 'few_shot')
    test_count = sum(1 for item in combined_data if item['category'] == category and item['data_type'] == 'test')
    print(f"{category}: {few_shot_count} few-shot, {test_count} test")