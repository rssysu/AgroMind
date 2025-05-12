import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from PIL import Image

class DeepseekVL2ClientSmall:
    def __init__(self, model_name="deepseek-ai/deepseek-vl2-tiny", prompt=None):
        self.model_name = model_name
        self.prompt = prompt
        
        # Initialize model and processor
        self.vl_processor = DeepseekVLV2Processor.from_pretrained(model_name)
        self.tokenizer = self.vl_processor.tokenizer
        
        self.vl_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.vl_model = self.vl_model.to(torch.bfloat16).cuda().eval()
        
        self.clear_contents()
    
    def clear_contents(self):
        self.image_paths = []
        self.contents = []
        
        if self.prompt:
            self.contents.append(self.prompt)
    
    def add_message(self, message):
        self.contents.append(message)
    
    def add_image(self, image_path):
        self.image_paths.append(image_path)
    
    def change_prompt(self, new_prompt):
        """Change system prompt"""
        self.prompt = new_prompt
        self.clear_contents()
    
    def get_response(self):
        # Prepare conversation format
        user_content = ""
        
        # Add <image> tag for each image
        for i, _ in enumerate(self.image_paths):
            user_content += f"<image>\n"
        
        # Add text content
        if self.contents:
            user_content += " ".join(self.contents)
        else:
            user_content += "Can you tell me what are in the images?"
        
        conversation = [
            {
                "role": "<|User|>",
                "content": user_content,
                "images": self.image_paths,
            },
            {"role": "<|Assistant|>", "content": ""}
        ]
        
        # Process input
        with torch.no_grad():
            # Load images
            pil_images = load_pil_images(conversation)
            
            # Prepare input
            prepare_inputs = self.vl_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(self.vl_model.device)
            
            # Run image encoder to get image embeddings
            inputs_embeds = self.vl_model.prepare_inputs_embeds(**prepare_inputs)
            
            # incremental_prefilling
            inputs_embeds, past_key_values = self.vl_model.incremental_prefilling(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                chunk_size=512  # prefilling size
            )
            
            # Run model to get response
            outputs = self.vl_model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,
                
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                
                do_sample=False,
                use_cache=True,
            )
            
            answer = self.tokenizer.decode(
                outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
                skip_special_tokens=False
            )
            answer = answer.replace("<｜end▁of▁sentence｜>", "").strip()
            
        
        return answer