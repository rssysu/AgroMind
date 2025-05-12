import torch
from transformers import AutoModel, AutoTokenizer

class XComposerClient:
    def __init__(self, model_name='internlm/internlm-xcomposer2-4khd-7b', prompt=None, 
                hd_num=45, do_sample=False, num_beams=3):
        torch.set_grad_enabled(False)
        
        # Initialize model and tokenizer
        self.model = AutoModel.from_pretrained(model_name, 
                                              torch_dtype=torch.bfloat16, 
                                              trust_remote_code=True).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Model parameters
        self.prompt = prompt
        self.hd_num = hd_num
        self.do_sample = do_sample
        self.num_beams = num_beams
        
        # Conversation state
        self.history = []
        self.current_query = ""
        self.current_image = None
        self.clear_contents()
    
    def clear_contents(self):
        self.history = []
        self.current_image = None
        if self.prompt:
            self.current_query = self.prompt
        else:
            self.current_query = ""

    
    def add_message(self, message):
        if self.current_query == "":
            self.current_query = message
        else:
            self.current_query = self.current_query + " " + message
    
    def add_image(self, image_path):
        self.current_image = image_path
        
        # If <ImageHere> tag is not in the query and there is already query text, add it automatically
        if self.current_query and '<ImageHere>' not in self.current_query:
            self.current_query = '<ImageHere>' + self.current_query
    
    def change_prompt(self, new_prompt):
        self.prompt = new_prompt
        self.clear_contents()
    
    def get_response(self):
        # Ensure the query contains <ImageHere> tag (when there is an image)
        query = self.current_query
        if self.current_image and '<ImageHere>' not in query:
            query = '<ImageHere>' + query
            
        with torch.cuda.amp.autocast():
            response, new_history = self.model.chat(
                self.tokenizer, 
                query=query,
                image=self.current_image,
                history=self.history,
                do_sample=self.do_sample,
                num_beams=self.num_beams
            )
            self.history = new_history
            
        return response