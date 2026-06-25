import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    # def __getitem__(self, index):
    #     line = self.questions[index]
    #     image_file = line["image"]
    #     qs = line["text"]
    #     if self.model_config.mm_use_im_start_end:
    #         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    #     conv = conv_templates[args.conv_mode].copy()
    #     conv.append_message(conv.roles[0], qs)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()

    #     image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
    #     image_tensor = process_images([image], self.image_processor, self.model_config)[0]

    #     input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    #     return input_ids, image_tensor, image.size

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["text"]
        
        images = []
        q_type = line.get("question_type_id") or line.get("task_type_id")
        qid = line.get("question_id")
        
        if q_type in [10, 11, 12] or str(q_type) in ["10", "11", "12"]:
            print(f"[DEBUG] Video question - qid: {qid}, q_type: {q_type}")  # 调试用
            
            base_dir = os.path.join(self.image_folder, "SEED-Bench-video-image", "v1_video")
            task_map = {
                10: "ssv2_8_frame",
                11: "kitchen_8_frame",
                12: "breakfast_8_frame"
            }
            subdir = task_map.get(int(q_type))
            
            if subdir:
                frame_dir = os.path.join(base_dir, f"task{q_type}", subdir, str(qid))
                print(f"[DEBUG] Looking for frames in: {frame_dir}")  # 关键调试
                
                for i in range(1, 9):
                    frame_path = os.path.join(frame_dir, f"{i}.png")
                    if os.path.exists(frame_path):
                        images.append(Image.open(frame_path).convert('RGB'))
                    # else:
                    #     print(f"[Warning] Missing {frame_path}")  # 可临时打开
                
        else:
            # 普通图像
            image_file = line["image"]
            image_path = os.path.join(self.image_folder, image_file)
            if os.path.exists(image_path):
                images.append(Image.open(image_path).convert('RGB'))
        
        if not images:
            print(f"[ERROR] No images found for qid: {qid} (type {q_type}). Checked dir: {frame_dir if 'frame_dir' in locals() else 'N/A'}")
            raise ValueError(f"No images found for question_id: {qid}")
        
        # 后续代码不变（多帧提示词 + 处理）
        num_images = len(images)
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * num_images + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN * num_images + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images(images, self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        image_sizes = [img.size for img in images]
        return input_ids, image_tensor, image_sizes

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
