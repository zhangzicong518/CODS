import json
import os
import random
from collections import defaultdict

data = json.load(open("/home/zicong/CODS/evaluate/few_shot_for_gradient.json", "r"))

# Create output directory
output_dir = "/home/zicong/CODS/evaluate/single"
os.makedirs(output_dir, exist_ok=True)

# Group data by category
all_Cate = set()
category_data = defaultdict(list)
for item in data:
    category = item['metadata']['category']
    category_data[category].append(item)
    all_Cate.add(category)
# Print all categories found
print(list(all_Cate))
exit(0)
# Sample 5 items from each category and save
for category, items in category_data.items():
    # Sample 5 items (or all if less than 5)
    sampled_items = random.sample(items, min(5, len(items)))
    
    # Create filename (replace spaces/special chars with underscores)
    filename = f"{category.replace(' ', '_').replace('/', '_')}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(sampled_items, f, indent=2)
    
    print(f"Saved {len(sampled_items)} items for category '{category}' to {filename}")

print(f"\nAll category files saved to: {output_dir}")