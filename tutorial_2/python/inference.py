import torch
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor


class Inference:
    def __init__(self):
        self.model = ""
        self.processor = ""
        self.inputs = ""

    def load_model(self):
        #Load the Grounding DINO model from HuggingFace
        print("Loading Grounding DINO model...")
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = processor = GroundingDinoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_id)
        self.model.eval()

    def preprocess(self,image,text_prompt):
        #Prepare inputs and run inference
        self.inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**self.inputs)
        
        return outputs
