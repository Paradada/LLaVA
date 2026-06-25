import torch
import torch.nn.functional as F
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX


def load_model(model_path):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        "llava-v1.5-7b",
        device_map="auto"
    )
    model.eval()
    return tokenizer, model, image_processor


def prepare_inputs(image_path, prompt, tokenizer, image_processor, model):
    image = Image.open(image_path).convert("RGB")

    image_tensor = process_images([image], image_processor, model.config)[0]

    qs = prompt
    qs = "<image>\n" + qs

    input_ids = tokenizer_image_token(
        qs,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).cuda()

    image_tensor = image_tensor.unsqueeze(0).half().cuda()

    return input_ids, image_tensor


@torch.no_grad()
def get_logits(model, input_ids, image_tensor):
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        output_hidden_states=False,
        return_dict=True
    )
    return outputs.logits


def compare_logits(logits_a, logits_b, topk=10):
    diff = torch.norm(logits_a - logits_b).item()
    print("L2 diff:", diff)

    prob_a = F.softmax(logits_a[:, -1, :], dim=-1)
    prob_b = F.softmax(logits_b[:, -1, :], dim=-1)

    top_a = torch.topk(prob_a, topk)
    top_b = torch.topk(prob_b, topk)

    print("\nTop tokens A:")
    print(top_a.indices[0].tolist())

    print("\nTop tokens B:")
    print(top_b.indices[0].tolist())


def run(model_path_a, model_path_b, image_path, question):
    tok_a, model_a, proc_a = load_model(model_path_a)
    tok_b, model_b, proc_b = load_model(model_path_b)

    input_ids_a, img_a = prepare_inputs(image_path, question, tok_a, proc_a, model_a)
    input_ids_b, img_b = prepare_inputs(image_path, question, tok_b, proc_b, model_b)

    logits_a = get_logits(model_a, input_ids_a, img_a)
    logits_b = get_logits(model_b, input_ids_b, img_b)

    compare_logits(logits_a, logits_b)


if __name__ == "__main__":
    image_path = "test.jpg"
    question = "What is in the image?"

    run(
        "YOUR_OFFICIAL_LORA_PATH",
        "YOUR_CUSTOM_LORA_PATH",
        image_path,
        question
    )