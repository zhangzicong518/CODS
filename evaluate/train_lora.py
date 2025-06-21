import argparse
import subprocess
import time
import os
import threading
from queue import Queue

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA with different methods")
    
    parser.add_argument("--method", type=str, required=True,
                       choices=["random", "repeat", "cods", "tive", "coincide"], 
                       help="Method to use: 'random', 'repeat', 'cods', or 'grad'")
   
    return parser.parse_args()

all_cates = ['logicocr', 'science_qa', 'viewspatial']


def run_training(gpu_id, train_data_path, output_dir, log_file, cate):
    """Run the swift sft training command on a specific GPU"""
    cmd = [
        'swift', 'sft',
        '--model', 'llava-hf/llava-1.5-7b-hf',
        '--train_type', 'lora',
        '--dataset', train_data_path,
        '--num_train_epochs', '3',
        '--per_device_train_batch_size', '4',
        '--learning_rate', '1e-5',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--gradient_accumulation_steps', '16',
        '--eval_steps', '10',
        '--save_steps', '10',
        '--save_total_limit', '3',
        '--output_dir', output_dir
    ]
    
    # Set environment for this specific GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Starting training for {cate}")
    print(f"[GPU {gpu_id}] Command: CUDA_VISIBLE_DEVICES={gpu_id} " + " ".join(cmd))
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ? Training completed successfully for {cate}")
        else:
            print(f"[GPU {gpu_id}] ? Training failed for {cate}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"[GPU {gpu_id}] ? Error training {cate}: {e}")
        with open(log_file, 'a') as f:
            f.write(f"\nError: {str(e)}\n")
        return False

def worker(gpu_id, task_queue, method):
    """Worker function for each GPU"""
    while True:
        try:
            cate = task_queue.get(timeout=1)
        except:
            break
        
        train_data = f"/home/zicong/CODS/data/{method}/{cate}.json"
        output_dir = f"/home/zicong/CODS/models/{method}/{cate}"
        log_file = f"./logs/{method}_{cate}_train.log"
        
        # Check if training data exists
        if not os.path.exists(train_data):
            print(f"[GPU {gpu_id}] Training data not found: {train_data}")
            task_queue.task_done()
            continue
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run training
        success = run_training(gpu_id, train_data, output_dir, log_file, cate)
        
        task_queue.task_done()

def main():
    args = parse_args()
    print(f"Using method: {args.method}")
        
    # Create task queue
    task_queue = Queue()
    
    # Add all categories to the queue
    for cate in all_cates:
        task_queue.put(cate)
    
    # Create and start worker threads for each GPU
    threads = []
    gpu_ids = [1,2,3]
    
    for gpu_id in gpu_ids:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue, args.method))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Wait for all tasks to complete
    task_queue.join()
    
    print(f"\n{'='*60}")
    print("All categories processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()