import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

class IdeficsClient:
    def __init__(self, model_name: str = "HuggingFaceM4/idefics2-8b", prompt: str = None):
        # Set device
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).cuda().eval()
        
        self.prompt = prompt
        self.image = None
        self.message = ""
        self.clear_contents()
        
    def add_image(self, image_path: str) -> None:
        self.image = load_image(image_path)
        
    def add_message(self, message: str) -> None:
        if not self.message:
            self.message = message
        else:
            self.message = self.message + " " + message
        
    def clear_contents(self) -> None:
        self.image = None
        if self.prompt:
            self.message = self.prompt
        else:
            self.message = ""
    
    def change_prompt(self, new_prompt: str) -> None:
        self.prompt = new_prompt
        self.clear_contents()
    
    def get_response(self, max_new_tokens: int = 500) -> str:
        """Get model response"""
        if not self.message:
            raise ValueError("Prompt text is not set, please use the add_message method to add it.")
            
        if self.image is None:
            raise ValueError("Image not added, please use the add_image method to add an image.")
        
        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.message}
                ]
            }
        ]
        
        # Process input
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[self.image], return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate response
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if "Assistant:" in generated_text:
            assistant_response = generated_text.split("Assistant:", 1)[1].strip()
        
        return assistant_response