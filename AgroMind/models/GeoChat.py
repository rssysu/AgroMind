import torch
from PIL import Image
from geochat.conversation import conv_templates, Chat
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path

class GeoChatClient:
    def __init__(self, model_path: str = "MBZUAI/geochat-7B", device: str = "cuda:0", prompt: str = None):
        self.device = device
        self.model_path = model_path
        
        # Load model
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, False, False, device=device
        )
        
        self.model = self.model.eval()
        
        # Initialize chat
        self.chat = Chat(self.model, self.image_processor, self.tokenizer, device=device)
        
        # Initialize state
        self.prompt = prompt
        self.image = None
        self.message = ""
        self.clear_contents()
        
    def add_image(self, image_path: str) -> None:
        if isinstance(image_path, str):
            self.image = Image.open(image_path).convert('RGB')
        else:
            self.image = image_path
            
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
        
    def get_response(self, temperature: float = 0.6, max_new_tokens: int = 300) -> str:
        if not self.message:
            raise ValueError("Prompt text is not set, please use the add_message method to add it.")
            
        if self.image is None:
            raise ValueError("Image not added, please use the add_image method to add an image.")
        
        # Create chat state
        chat_state = conv_templates['llava_v1'].copy()
        img_list = []
        
        # Upload image
        self.chat.upload_img(self.image, chat_state, img_list)
        
        # Ask question
        self.chat.ask(self.message, chat_state)
        
        # Get answer
        if len(img_list) > 0:
            if not isinstance(img_list[0], torch.Tensor):
                self.chat.encode_img(img_list)
                
        # Stream answer generation
        streamer = self.chat.stream_answer(
            conv=chat_state,
            img_list=img_list,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000
        )
        
        # Collect full answer
        output = ''
        for new_output in streamer:
            output += new_output
            
        return output