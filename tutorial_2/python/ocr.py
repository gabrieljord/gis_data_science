import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image, ImageOps
import re

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "Qwen/Qwen2-VL-2B-Instruct"

# Initialize model and processor once to avoid reloading on every inference
print(f"Loading {model_id} for OCR...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype
).to(device)
processor = AutoProcessor.from_pretrained(model_id)
print("OCR Model loaded.")

def extract_text(cropped_pil_image):
    """
    Takes a cropped PIL Image and performs OCR using Qwen2-VL.
    Handles vertically-stacked text by padding to square and prompting carefully.
    """
    if cropped_pil_image.mode != "RGB":
        cropped_pil_image = cropped_pil_image.convert("RGB")
        
    # Pad the crop to a square. Qwen2-VL works much better when it has square context 
    # instead of an extremely thin vertical sliver where letters look distorted.
    width, height = cropped_pil_image.size
    new_size = max(width, height)
    delta_w = new_size - width
    delta_h = new_size - height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    padded_img = ImageOps.expand(cropped_pil_image, padding, fill=(255, 255, 255))

    # DEBUG: Save the padded crop
    padded_img.save("debug_padded_crop.png")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is the vertical text written on the yellow tag in the center? Please output the characters from top to bottom."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text], 
        images=[padded_img], 
        padding=True, 
        return_tensors="pt"
    ).to(device, torch_dtype)
    
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0].strip()
    
    # Strip any extra quotes or periods it might have added
    output_text = re.sub(r'[^A-Za-z0-9]', '', output_text)
    
    return output_text
