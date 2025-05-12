import torch
from PIL import Image
from io import BytesIO
import requests
from typing import List, Union, Dict, Any, Optional

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


class MantisIdefics2Client:
    def __init__(self, model_name: str = "TIGER-Lab/Mantis-8B-Idefics2", prompt: str = None):
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto"
        )
        
        # Default generation parameters
        self.generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False
        }
        
        self.prompt = prompt
        self.images = []
        self.messages = []
        self.clear_contents()
        
    def add_image(self, image_path: str) -> None:
        image = load_image(image_path)
        self.images.append(image)
        
    def add_message(self, message: str) -> None:
        if not self.messages:
            self.messages = message
        else:
            self.messages = self.messages + " " + message
        
    def clear_contents(self) -> None:
        self.images = []
        if self.prompt:
            self.messages = self.prompt
        else:
            self.messages = ""
    
    def change_prompt(self, new_prompt: str) -> None:
        self.prompt = new_prompt
        self.clear_contents()
    
    def get_response(self) -> str:
        if not self.messages:
            raise ValueError("Prompt text is not set, please use the add_message method to add it.")
            
        if not self.images:
            raise ValueError("Image not added, please use the add_image method to add an image.")
        
        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in range(len(self.images))),
                    {"type": "text", "text": self.messages},
                ]
            }    
        ]
        
        # Apply chat template and process input
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=self.images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        generated_ids = self.model.generate(**inputs, **self.generation_kwargs)
        response = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response[0]