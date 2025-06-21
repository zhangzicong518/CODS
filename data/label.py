import os
import base64
import json
import time
from PIL import Image
from io import BytesIO
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime, timedelta
from model import Qwen2_5_VL

model = Qwen2_5_VL()

def encode_image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class RateLimiter:
    """请求频率限制器"""

    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests  # 最大请求数
        self.time_window = time_window  # 时间窗口（秒）
        self.requests = []  # 请求时间记录
        self.lock = Lock()  # 线程锁

    def acquire(self):
        """获取请求许可"""
        with self.lock:
            now = datetime.now()
            # 清理过期的请求记录
            self.requests = [req_time for req_time in self.requests if now - req_time < timedelta(seconds=self.time_window)]

            # 如果当前请求数达到上限，等待
            if len(self.requests) >= self.max_requests:
                oldest = min(self.requests)
                sleep_time = (oldest + timedelta(seconds=self.time_window) - now).total_seconds()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 记录新的请求时间
            self.requests.append(now)


def process_single_data(args):
    image_file, json_file, question, answer, rate_limiter = args

    rate_limiter.acquire()

    label = model.inference(
        image_path=image_file,
        ques=question,
    )

    return label

def extract_info_from_data(data):
    """
    从JSON数据中提取问题和答案信息
    
    :param data: 包含conversations的JSON数据
    :return: 元组 (question_string, answer_string)
    """
    question = ""
    answer = ""
    
    # 提取问题（来自human）
    if "conversations" in data:
        human_messages = [msg for msg in data["conversations"] if msg.get("from") == "human"]
        if human_messages:
            question_text = human_messages[0].get("value", "")
            
            # 去掉<image>标记
            question_text = question_text.replace("<image>", "").strip()
            
            # 去掉Hint开头的提示信息
            import re
            question_text = re.sub(r'^Hint:.*?\n', '', question_text, flags=re.DOTALL)
            
            # 去掉"Please answer..."等提示语
            question_text = re.sub(r'Please answer the question and provide the correct option letter.*?\n', '', question_text)
            
            # 提取问题部分
            question = question_text.strip()
        
        # 提取答案（来自gpt）
        gpt_messages = [msg for msg in data["conversations"] if msg.get("from") == "gpt"]
        if gpt_messages:
            answer_text = gpt_messages[0].get("value", "")
            
            # 提取选项字母（A、B、C、D等）
            option_match = re.search(r'[Tt]he answer is ([A-Za-z])', answer_text)
            if option_match:
                answer = option_match.group(1).upper()
            else:
                # 如果没有明确格式，保留整个答案
                answer = answer_text.strip()
    
    return question, answer
   

def process_data(directory_path, output_path=None, max_images=None, max_workers=5):
    """
    使用多线程处理目录下的所有图片
    :param directory_path: 图片目录路径
    :param api_key: API密钥
    :param processed_files: 已处理的文件集合
    :param max_images: 最大处理图片数量
    :param max_workers: 最大线程数
    :return: 图片文件名和识别结果的字典
    """
    existing_data_jsonl = output_path
    existing_data = []
    with open(existing_data_jsonl, 'r') as f:
        for line in f:
            existing_data.append(json.loads(line.strip()))

    if len(existing_data) == 0:
        last_idx = -1
    else:
        last_idx = existing_data[-1]["id"]

    all_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.json'))]

    # 创建频率限制器（每分钟最多1200次请求）
    rate_limiter = RateLimiter(max_requests=1000, time_window=60)

    tasks = []
    for json_file in all_files:
        try:
            with open(os.path.join(directory_path, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            (question, answer) = extract_info_from_data(data)
            json_file = os.path.join(directory_path, json_file)
            image_file = os.path.join(directory_path, json_file.replace('.json', '.jpg'))
            tasks.append((image_file, json_file, question, answer, rate_limiter))
        except Exception as e:
            print(f"处理 {json_file} 时发生错误: {str(e)}")
            continue

    pbar = tqdm(total=len(tasks), desc="处理进度")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_single_data, task): task for task in tasks}

        # 处理完成的任务
        for future in as_completed(future_to_file):
            task = future_to_file[future]
            try:
                label = future.result()
                if label is not None:
                    image_file, json_file, question, answer, rate_limiter = task
                    new_item = {"id": last_idx + 1}
                    last_idx += 1
                    print(f"{last_idx} finished")
                    new_item["source"] = "Infinity-MM"
                    new_item["source_id"] = image_file
                    
                    img_path = "your/image/path"  # 替换为实际的图片存储路径
                    # if path not exists, create it
                    if not os.path.exists(img_path):
                        os.makedirs(img_path)
                    
                    new_item["question"] = question
                    new_item["label"] = answer
                    
                    new_item["img_path"] = image_file
                    new_item["category"] = label
                    with open(existing_data_jsonl, 'a') as f:
                        f.write(json.dumps(new_item) + "\n") 
            except Exception as e:
                print(f"\n处理 {image_file} 时发生错误: {str(e)}")
            finally:
                pbar.update(1)

    pbar.close()

    return results


def main():
    # 设置参数
    image_directory = "your_image_directory"  # 替换为实际的图片目录路径
    output_file = "your_output_file.jsonl"  # 替换为实际的输出文件路径

    max_images = None  # 设置为None或者具体数字
    max_workers = 1  # 并发线程数，可以根据需要调整

    process_data(
        directory_path=image_directory, output_path=output_file, max_images=max_images, max_workers=max_workers
    )


if __name__ == "__main__":
    main()
