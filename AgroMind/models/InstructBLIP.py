from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import os
import time

class InstructBLIPClient:
    def __init__(self, model_name="Salesforce/instructblip-vicuna-7b", prompt=None):
        # Load model and processor
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_name).cuda().eval()
        self.processor = InstructBlipProcessor.from_pretrained(model_name)
        
        # Initialize content
        self.contents = ""
        self.image = None
        self.prompt = prompt
        self.clear_contents()
    
    def add_image(self, image_path):
        self.image = Image.open(image_path).convert("RGB")
    
    def add_message(self, message):
        if self.contents == "":
            self.contents = message
        else:
            self.contents = self.contents + " " + message
    
    def clear_contents(self):
        self.image = None
        if self.prompt:
            self.contents = self.prompt
        else:
            self.contents = ""
    
    def change_prompt(self, new_prompt):
        self.prompt = new_prompt
        self.clear_contents()
    
    def get_response(self, 
                     do_sample=False,
                     num_beams=5,
                     max_new_tokens=256,
                     min_length=1,
                     top_p=0.9,
                     repetition_penalty=1.5,
                     length_penalty=1.0,
                     temperature=1):
        if not self.contents:
            raise ValueError("Prompt text is not set, please use the add_message method to add it.")
            
        if self.image is None:
            raise ValueError("Image not added, please use the add_image method to add an image.")
        
        # Process image and text
        inputs = self.processor(images=self.image, text=self.contents, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate output
        outputs = self.model.generate(
            **inputs,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        
        # Decode output
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        if generated_text.startswith(self.contents):
            # If the output starts with the original prompt, remove this part
            generated_text = generated_text[len(self.contents):].strip()
        
        return generated_text