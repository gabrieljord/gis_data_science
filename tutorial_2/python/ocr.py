import easyocr
import numpy as np
from PIL import ImageEnhance

# Initialize the reader once
reader = easyocr.Reader(['en'], gpu=False)

def extract_text(cropped_pil_image):
    """
    Takes a cropped PIL Image and performs OCR using EasyOCR.
    Handles both horizontal and vertically-stacked text.
    """
    # 1. Upscale the image slightly if it's very small
    width, height = cropped_pil_image.size
    if width < 150 or height < 150:
        # Use high quality LANCZOS resampling to reduce blur
        cropped_pil_image = cropped_pil_image.resize((width * 3, height * 3), resample=3)
        
    # User requested: Rotate the image 90 degrees counter-clockwise
    cropped_pil_image = cropped_pil_image.rotate(90, expand=True)

    # Enhance sharpness and contrast for the blurry text
    enhancer = ImageEnhance.Sharpness(cropped_pil_image)
    cropped_pil_image = enhancer.enhance(3.0) # Sharpen aggressively
    
    enhancer_cont = ImageEnhance.Contrast(cropped_pil_image)
    cropped_pil_image = enhancer_cont.enhance(1.5) # Slight contrast boost

    # DEBUG: Save the cropped and upscaled image to disk
    cropped_pil_image.save("debug_crop.png")

    img_array = np.array(cropped_pil_image)
    
    # 2. Read text using EasyOCR
    # Add rotation_info so EasyOCR actively tries to read it sideways and upside down too, 
    # taking the highest confidence result.
    results = reader.readtext(
        img_array,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        rotation_info=[90, 180, 270],
        text_threshold=0.1,
        low_text=0.1,
        mag_ratio=1.5
    )
    
    if not results:
        return ""

    # 3. Determine if the tag is vertical or horizontal based on its aspect ratio
    is_vertical = height > width

    text_data = []
    for bbox, text, conf in results:
        # Calculate the center (x, y) of each detected text box
        center_x = sum([pt[0] for pt in bbox]) / 4
        center_y = sum([pt[1] for pt in bbox]) / 4
        text_data.append((center_x, center_y, text))
    
    # 4. Sort the text intelligently
    # Because we rotated the image 90 degrees CCW, what used to be top-to-bottom
    # is now left-to-right! So we sort by the X coordinate.
    text_data.sort(key=lambda item: (round(item[0] / 5), item[1]))

    # Join the sorted text fragments
    detected_texts = [item[2] for item in text_data]
    return "".join(detected_texts)
