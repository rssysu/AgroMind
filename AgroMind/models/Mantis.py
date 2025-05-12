import torch
from PIL import Image
from mantis.models.mllava import chat_mllava, MLlavaProcessor, LlavaForConditionalGeneration


class MantisClient:
    def __init__(self, model_name: str = "TIGER-Lab/Mantis-8B-siglip-llama3", prompt: str = None, 
                 device: str = "cuda", attn_implementation: str = None):
        # Load processor and model
        self.processor = MLlavaProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            device_map=device, 
            torch_dtype=torch.bfloat16, 
            attn_implementation=attn_implementation
        )
        
        # Generation parameters
        self.generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False
        }
        
        self.prompt = prompt
        self.images = []
        self.messages = ""
        self.clear_contents()
        
    def add_image(self, image_path: str) -> None:
        self.images.append(Image.open(image_path).convert("RGB"))
        
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
        
        # Call the model to generate response
        response, history = chat_mllava(
            self.messages, 
            self.images, 
            self.model, 
            self.processor, 
            **self.generation_kwargs
        )
        
        return response