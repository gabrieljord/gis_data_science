from ultralytics import YOLOWorld

# 1. Load the pre-trained zero-shot model
# Note: 'v2' models support exporting to edge devices later
model = YOLOWorld('yolov8x-worldv2.pt')

# 2. Tell the LLM exactly what to search for using text prompts
model.set_classes(["utility_pole","yellow tag", "silver metal plate", "sticker",     
  "wooden pole"])

# 3. Run inference on your image (lowering confidence threshold to find more objects)
results = model.predict("../data/images/pole.png", conf=0.07)

# 4. Show and save the detected objects
results[0].show()
results[0].save(filename="prediction_result.png")
print("Results saved to prediction_result.png")
print(results[0])