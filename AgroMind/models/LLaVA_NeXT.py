from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

class LLaVANeXTClient:
    def __init__(self, model_name="llava-hf/llava-v1.6-34b-hf", prompt=None):
        # Load processor
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        
        # Load model
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).cuda().eval()
        
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
    
    def get_response(self, max_new_tokens=100):
        if not self.contents:
            raise ValueError("Prompt text is not set, please use the add_message method to add it.")
            
        if not self.image_path:
            raise ValueError("Image not added, please use the add_image method to add an image.")
        
        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image_path},  # Local path or URL
                    {"type": "text", "text": self.contents},
                ],
            },
        ]
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )

        # Move inputs to GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode response
        response_text = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract the answer part from the model output, remove instruction tags and question
        if "[/INST]" in response_text:
            # Only keep the content after [/INST]
            clean_response = response_text.split("[/INST]", 1)[1].strip()
        elif "ASSISTANT:" in response_text:
            clean_response = response_text.split("ASSISTANT:", 1)[1].strip()
        elif "assistant" in response_text:
            clean_response = response_text.split("assistant", 1)[1].strip()
        else:
            clean_response = response_text
    
        return clean_response