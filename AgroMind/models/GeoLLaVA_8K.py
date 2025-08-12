import torch
from PIL import Image
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX

class GeoLLaVAClient:
    def __init__(self, model_name="initiacms/GeoLLaVA-8K", prompt=None):
        # Fix seed
        torch.manual_seed(0)
        
        self.model_name = model_name
        self.max_frames_num = 16
        
        # Generation kwargs
        self.gen_kwargs = {
            "do_sample": False, 
            "top_p": None, 
            "num_beams": 1, 
            "use_cache": True, 
            "max_new_tokens": 1024
        }
        
        # Load model components
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_name, None, "llava_qwen", device_map="cuda:0"
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
        
        # Prepare prompt
        if "<image>" not in self.contents:
            prompt = f"<|im_start|>system\nFollow the instructions to answer the question<|im_end|>\n<|im_start|>user\n<image>\n{self.contents}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = self.contents
        
        # Prepare input
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        
        image = Image.open(self.image_path).convert("RGB")
        images_tensor = process_images([image], self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, 
                images=images_tensor, 
                image_sizes=[image.size], 
                modalities=["image"], 
                **self.gen_kwargs
            )
        
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
