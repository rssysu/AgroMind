import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

class DeepseekVL2Client:
    def __init__(self, model_name="deepseek-ai/deepseek-vl2-tiny", prompt=None):
        # Load processor
        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_name)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        # Load model
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        
        # Initialize conversation content
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
        
        # Build conversation format
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{self.contents}",
                "images": [self.image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Load image and prepare input
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)
        
        # Run image encoder to get image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # Run model to generate response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        
        # Decode output to get answer
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # Extract the actual answer part from the model
        if "<|Assistant|>" in answer:
            answer = answer.split("<|Assistant|>")[-1].strip()
        
        return answer