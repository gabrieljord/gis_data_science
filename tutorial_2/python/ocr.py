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
    if cropped_pil_image.mode != "RGB":
        cropped_pil_image = cropped_pil_image.convert("RGB")
        
    # The cropped image is a very thin sliver. We stretch the image horizontally 
    # so it's wider (width = height // 2). This makes the characters look fat, 
    # but gives the model plenty of pixels to look at!
    width, height = cropped_pil_image.size
    new_width = height // 2
    stretched_img = cropped_pil_image.resize((new_width, height), Image.Resampling.LANCZOS)
    
    # Save debug image
    stretched_img.save("debug_qwen_stretched.png")

    # STRICT CONSTRAINED PROMPT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "This is a utility pole serial number stacked vertically. The format is EXACTLY 2 uppercase letters followed by 4 digits. Extract the 6-character serial number. Output ONLY the 6 alphanumeric characters."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text], 
        images=[stretched_img], 
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
    
    print(f"  *** Raw Qwen output: {output_text}")
    
    # Strip everything except alphanumeric characters
    output_text = re.sub(r'[^A-Za-z0-9]', '', output_text)
    
    return output_text
