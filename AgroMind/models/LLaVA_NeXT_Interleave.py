from lmdeploy import pipeline, GenerationConfig
from typing import List, Dict, Union, Optional

class LLaVANextInterleaveClient:
    def __init__(self, model_name: str = "llava-hf/llava-interleave-qwen-7b-hf", prompt: str = None):
        # Load model
        self.pipe = pipeline(model_name)
        
        self.prompt = prompt
        self.images = []
        self.messages = []
        self.clear_contents()
        
    def add_image(self, image_path: str) -> None:
        self.images.append(dict(type='image', image=image_path))
        
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
        content = [dict(type='text', text=self.messages)] + self.images
        formatted_messages = [dict(role='user', content=content)]
        
        # Configure generation parameters
        config = GenerationConfig(top_k=1)
        
        # Call the model to generate response
        response = self.pipe(formatted_messages, gen_config=config)
        
        return response.text