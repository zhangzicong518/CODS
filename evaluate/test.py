from datasets import load_from_disk, Dataset
from collections import defaultdict
import argparse
import os
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
import tempfile
from PIL import Image
import torch
import gc
import transformers
import warnings
import json

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Test MMBench with different methods")
    
    parser.add_argument("--method", type=str, required=True,
                       choices=["pure", "repeat", "random", "cods", "coincide", "tive"], 
                       help="Method to use: 'pure' for base LLaVA-v1.5, others are lora ones")
   
    return parser.parse_args()
args = parse_args()

class llava_v1_5():
    
    def __init__(self, model_path = None, lora_ckpt = None):
        
        if lora_ckpt:
            model, tokenizer = get_model_tokenizer("llava-hf/llava-1.5-7b-hf")
            lora_checkpoint = safe_snapshot_download(lora_ckpt)
            model = Swift.from_pretrained(model, lora_ckpt)
        else:
            model, tokenizer = get_model_tokenizer(model_path, model_type="llava1_5_hf")
            model = Swift.from_pretrained(model, model_path, model_type="llava1_5_hf")
        template_type = None
        template_type = template_type or model.model_meta.template
        template = get_template(template_type, tokenizer)
        self.engine = PtEngine.from_model_template(model, template, max_batch_size=1)
        self.request_config = RequestConfig(max_tokens=2048, temperature=0)

    def inference(self, image_path, prompt):
        
        infer_requests = [
            InferRequest(messages=[{'role': 'user', 'content': f"<image>{prompt}"}],
                        images=[image_path])
        ]
    
        resp_list = self.engine.infer(infer_requests, self.request_config)
        responses = [resp.choices[0].message.content for resp in resp_list]
        return responses[0]

def save_image_to_temp(image_obj):
    """Save PIL Image object to temporary file and return path"""
    temp_dir = "./tmp/mmbench_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.jpg', delete=False) as f:
        image_obj.save(f, format='JPEG')
        return f.name

def load_model_for_category(method, category):
    """
    Load appropriate model based on method and category
    """
    
    torch.cuda.empty_cache()
    gc.collect()

    if method == "pure":
        print(f"  Loading base model...")
        model = llava_v1_5(model_path="/home/zicong/.cache/modelscope/hub/models/llava-hf/llava-1.5-7b-hf")
        return model
        
    else:
        print(f"  Loading LoRA model for {category}...")
        
        lora_ckpt = f"/home/zicong/CODS/models/{args.method}/{category}/"
        # find folders in loar_ckpt in v0
        lora_folders = [f.path for f in os.scandir(lora_ckpt) if f.is_dir()]
        lora_ckpt = lora_folders[0] if lora_folders else None
        lora_ckpt = os.path.join(lora_ckpt, "checkpoint-90")
        if not os.path.exists(lora_ckpt):
            lora_ckpt = lora_ckpt.replace("checkpoint-90", "checkpoint-87")
        model = llava_v1_5(model_path="/home/zicong/.cache/modelscope/hub/models/llava-hf/llava-1.5-7b-hf", lora_ckpt=lora_ckpt)
        
        return model

def test_model(model, image, question):
    """
    Test model with image and question
    """
    response = model.inference(image, question)
    response = response.strip().upper()
    if response in ['A', 'B', 'C', 'D']:
        predicted_answer = response
    else:
        # Try to find A, B, C, or D in the response
        for option in ['A', 'B', 'C', 'D']:
            if option in response:
                predicted_answer = option
                break
        else:
            predicted_answer = 'Q'  # Default fallback
    
    return predicted_answer

all_cates = ['logicocr', 'science_qa', 'viewspatial']

def main():

    print("Dataset Overview:")

    # Initialize results storage
    category_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'accuracy': 0.0})
    test_results = []
    
    # Test items category by category
    print(f"\nTesting items by category using {args.method} method...")
    
    current_model = None

    for category in all_cates:
        
        print(f"\nProcessing category: {category}")
        
        data = json.load(open(f"/home/zicong/DATASETS/test_expansion/test/{category}.json", 'r'))    
        
        if current_model is not None:
            
            del current_model
            current_model = None
        
        print(f"  Loading new model for category: {category}")
        current_model = load_model_for_category(args.method, category)
        
        for i, item in enumerate(data):
            question = item['question']
            image = item['image']
            correct_answer = item['answer']
      
            # Test with model
            predicted_answer = test_model(current_model, image, question)
            
            # Check if correct
            is_correct = predicted_answer == correct_answer
            
            # Store individual result
            test_result = {
                'question_id': i,
                'response': predicted_answer,
                'answer': correct_answer,
                'category': category,
                'question': question,
                'is_correct': is_correct,
            }
            test_results.append(test_result)
            
            # Update category stats
            category_results[category]['total'] += 1
            if is_correct:
                category_results[category]['correct'] += 1
            
        
        # Thoroughly clear model after processing category
        print(f"  Finished processing {category}")
        if current_model is not None:
            if hasattr(current_model, 'cleanup'):
                current_model.cleanup()
            del current_model
            current_model = None
        

    # Calculate accuracies and prepare category results
    category_stats = {}
    for category in category_results:
        total = category_results[category]['total']
        correct = category_results[category]['correct']
        accuracy = correct / total if total > 0 else 0.0
        category_stats[category] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
    
    # Calculate overall stats
    total_correct = sum(stats['correct'] for stats in category_stats.values())
    total_items = sum(stats['total'] for stats in category_stats.values())
    overall_accuracy = total_correct / total_items if total_items > 0 else 0.0
    
    # Prepare final JSON structure
    results_json = {
        'method': args.method,
        'overall': {
            'correct': total_correct,
            'total': total_items,
            'accuracy': overall_accuracy
        },
        'category_stats': category_stats,
        'questions': test_results
    }
    
    # Report results
    print("\n" + "="*80)
    print(f"TEST RESULTS BY CATEGORY ({args.method.upper()} METHOD):")
    print("="*80)
    print(f"{'Category':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-"*80)
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        correct = stats['correct']
        total = stats['total']
        accuracy = stats['accuracy']
        
        print(f"{category:<30} {correct:<10} {total:<10} {accuracy:.3f}")
    
    print("-"*80)
    print(f"{'OVERALL':<30} {total_correct:<10} {total_items:<10} {overall_accuracy:.3f}")
    
    print(f"\nOverall Test Accuracy: {overall_accuracy:.1%}")

if __name__ == "__main__":
    main()