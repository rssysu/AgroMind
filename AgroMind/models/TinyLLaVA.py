import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TinyLLaVAClient:
    def __init__(self, model_name="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B", prompt=None):
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=False,
            model_max_length=config.tokenizer_model_max_length, 
            padding_side=config.tokenizer_padding_side,
            trust_remote_code=True
        )
        
        self.contents = ""
        self.image_path = None
        self.prompt = prompt
        self.clear_contents()
        
    def add_image(self, image_path):
        self.image_path = image_path
        
    def add_message(self, message):
        if self.contents == "":
            self.contents = message
        else:
            self.contents = self.contents + " " + message
        
    def clear_contents(self):
        self.image_path = None
        if self.prompt:
            self.contents = self.prompt
        else:
            self.contents = ""
    
    def change_prompt(self, new_prompt):
        self.prompt = new_prompt
        self.clear_contents()
    
    def get_response(self):
        if not self.contents:
            raise ValueError("Prompt text is not set, please use the add_message method to add it.")
            
        if not self.image_path:
            raise ValueError("Image not added, please use the add_image method to add an image.")
            
        # Use the model's chat method to generate the answer
        output_text, generation_time = self.model.chat(
            prompt=self.contents,
            image=self.image_path,
            tokenizer=self.tokenizer
        )
        
        return output_text