import base64
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

class InternVLClient:
    def __init__(self, model_name="OpenGVLab/InternVL2-8B", prompt=None):
        self.model_name = model_name
        self.prompt = prompt
        self.images = []
        self.contents = []
        # Initialize model
        self.pipe = pipeline(self.model_name, 
                            backend_config=TurbomindEngineConfig(session_len=16384))
        self.clear_contents()
    
    def clear_contents(self):
        self.images = []
        if self.prompt:
            self.contents = [self.prompt]
        else:
            self.contents = []
    
    def add_message(self, message):
        self.contents.append(message)
    
    def add_image(self, image_path):
        image = load_image(image_path)
        self.images.append(image)
    
    def change_prompt(self, new_prompt):
        self.prompt = new_prompt
        self.clear_contents()
    
    def get_response(self):
        # Create tags for each image
        image_tags = []
        for i in range(len(self.images)):
            image_tags.append(f"Image-{i+1}: {IMAGE_TOKEN}")
        
        prompt_parts = []
        if image_tags:
            prompt_parts.extend(image_tags)
            
        prompt_parts.extend(self.contents)
        full_prompt = "\n".join(prompt_parts)
            
        # Call the model to get response
        response = self.pipe((full_prompt, self.images))
        return response.text