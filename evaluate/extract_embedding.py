import torch
import json
import numpy as np
# embeddings_file = "/home/zicong/CODS/data/llava/all_embeddings.pt"
# target_json_file = "/home/zicong/CODS/data/llava/target_data_preocessed.json"

# embeddings_data = torch.load(embeddings_file, map_location='cpu')
# all_embeddings = embeddings_data['embeddings']
    
# with open(target_json_file, 'r', encoding='utf-8') as f:
#     target_data = json.load(f)

# def get_embedding(data):
#     if isinstance(data, dict):
#         target_id = data['id']
#         index = next((i for i, item in enumerate(target_data) if item.get('id') == target_id), None)
#         if index is None:
#             return None
#     else:
#         return None
    
#     return all_embeddings[index].numpy()

# # test
# print(get_embedding(target_data[0]))

#############################  few_shot version #############################

embeddings_file = "/home/zicong/CODS/data/test_expansion/all_embeddings.pt"
target_json_file = "/home/zicong/DATASETS/test_expansion/few-shot/merged_few_shot.json"

tive_file = "/home/zicong/CODS/data/test_expansion/output_combined"
tive_emb = torch.load(tive_file, map_location='cpu')

coincide_file = "/home/zicong/CODS/data/test_expansion/coincide.npy"
coincide_emb = torch.from_numpy(np.load(coincide_file, allow_pickle=True)).float()

embeddings_data = torch.load(embeddings_file, map_location='cpu')
all_embeddings = embeddings_data['embeddings']

with open(target_json_file, 'r', encoding='utf-8') as f:
    target_data = json.load(f)

id_mapping = {}

for i in range(len(target_data)):
    # print(target_data[i]['image'])
    id_mapping[target_data[i]['image']] = i    
    
def get_embedding_few_shot(data):
    return all_embeddings[id_mapping[data["image"]]].numpy()

def get_embedding_tive(data):
    return tive_emb[id_mapping[data["image"]]].numpy()

def get_embedding_coincide(data):
    return tive_emb[id_mapping[data["image"]]].numpy()

# test
# print(get_embedding_few_shot(target_data[0]))