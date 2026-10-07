import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import inference as model


infre = model.Inference()
infre.load_model()

#Load the image
image_path = "../data/images/pole.png"
print(f"Processing image: {image_path}")
image = Image.open(image_path).convert("RGB")

#Define text prompts (must be lower case and separated by periods for Grounding DINO)
text_prompt = "utility pole . pole_tag ."
print(f"Searching for: {text_prompt}")

outputs = infre.preprocess(image=image,text_prompt=text_prompt)

#Post-process to extract bounding boxes
results = infre.processor.post_process_grounded_object_detection(
    outputs,
    infre.inputs.input_ids,
    threshold=0.2,        # Minimum confidence for a box
    text_threshold=0.25,  # Minimum confidence for text matching
    target_sizes=[image.size[::-1]]
)[0]

print("\n--- Detections ---")
#Visualize the results in a window!
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image)
ax.axis('off')

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = box.tolist()
    score = score.item()
    print(f"- Detected '{label}' with confidence {score:.3f}")
    
    # Draw rectangle
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], 
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    # Add label text
    ax.text(box[0], max(0, box[1] - 5), f"{label}: {score:.2f}", 
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

plt.title("Grounding DINO Detections")
plt.tight_layout()
plt.savefig("prediction_result_dino.png")
print("\nSaved image to prediction_result_dino.png")
plt.show()