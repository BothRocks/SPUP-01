import glob
import json
import os
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "microsoft/Florence-2-large"

image_folder = "images/"
output_file = "spup01.jsonl"


task = "<CAPTION>"
# task = "<DETAILED_CAPTION>"
# task = "<MORE_DETAILED_CAPTION>"


def fixed_get_imports(filename):
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


def get_model_processor(model_id, device, torch_dtype):
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def get_image_files(folder):
    if not os.path.exists(folder):
        print(f"Error: La carpeta '{folder}' no existe.")
        return []

    image_files = (
        glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
        + glob.glob(os.path.join(folder, "*.png"))
    )
    return image_files


model, processor = get_model_processor(model_id, device, torch_dtype)
image_files = get_image_files(image_folder)

results = []
for index, image_file in enumerate(image_files, 1):

    # 1
    image = Image.open(image_file)

    # 2
    inputs = processor(text=task, images=image, return_tensors="pt").to(
        device, torch_dtype
    )

    # 3
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
        early_stopping=False,
    )

    # 4
    gen_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # 5
    parsed_answer = processor.post_process_generation(
        gen_text, task=f"{task}", image_size=(image.width, image.height)
    )

    # 6
    value = parsed_answer.pop(task)
    parsed_answer["text"] = value
    parsed_answer["file_name"] = os.path.basename(image_file)

    print(
        f"{index}/{len(image_files)} {parsed_answer['file_name']}: {parsed_answer['text']}"
    )
    results.append(parsed_answer)

with open(output_file, "w") as jsonl_file:
    for line in results:
        jsonl_file.write(json.dumps(line) + "\n")
