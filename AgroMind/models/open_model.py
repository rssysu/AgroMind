import base64
from openai import OpenAI
import openai
import time

class OpenAIClient:
    def __init__(self, base_url, api_key, prompt=None, model="gpt-4o", temperature=0.7):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.contents = []
        self.clear_contents()

    def clear_contents(self):
        if self.prompt:
            self.contents = [{"role": "system", "content": self.prompt},
                             {"role": "user", "content": []}]
        else:
            self.contents = [{"role": "user", "content": []}]
    
    def add_message(self, message):
        content = {"type": "text", "text": message}
        self.contents[-1]["content"].append(content)

    def add_image(self, image_path):
        from PIL import Image
        import io
        
        # Open image and get size
        img = Image.open(image_path)
        width, height = img.size
        
        # Check if image exceeds 2048x2048
        max_size = 2048
        if width > max_size or height > max_size:
            # Calculate scaling ratio, keep aspect ratio
            ratio = min(max_size / width, max_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"Compress image: {width}x{height} → {new_width}x{new_height}")
            
            # Convert resized image to bytes
            buffer = io.BytesIO()
            
            # Determine image format
            if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
                img.save(buffer, format="JPEG")
                image_format = "jpeg"
            else:
                img.save(buffer, format="PNG")
                image_format = "png"
            
            # Convert image to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            # If image does not exceed max size, read file directly
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine image format
            if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
                image_format = "jpeg"
            else:
                image_format = "png"
        
        # Add image to content
        content = {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64," + image_base64}}
        self.contents[-1]["content"].append(content)
    
    def change_prompt(self, new_prompt):
        self.prompt = new_prompt
        self.clear_contents()

    def get_response(self):
        max_attempts = 20  # Maximum retry times
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Try to send request
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.contents,
                    temperature=self.temperature,
                )
                return completion.choices[0].message.content
            except openai.RateLimitError as e:
                # Catch rate limit error and retry after 30 seconds
                print(f"Rate limit error: {e}. Retrying in 30 seconds...")
                time.sleep(30)
            except Exception as e:
                error_str = str(e)
                # Check for 413 error
                if "413" in error_str or "Request Entity Too Large" in error_str or "500" in error_str:
                    print(f"Detected 413 error: Request body too large. Trying to compress images...")
                    
                    # Compress all images to 50% size
                    if self._compress_images_by_half():
                        attempt += 1
                        print(f"Images compressed (attempt {attempt}), retrying request...")
                    else:
                        return "Error: Unable to further compress images or images do not exist. Please try using smaller images."
                else:
                    # Catch other errors and retry after waiting
                    print(f"Unexpected error occurred: {e}")
                    time.sleep(30)
            
            attempt += 1
        
        return "Maximum number of attempts reached, request failed."

    def _compress_images_by_half(self):
        from PIL import Image
        import io
        
        compressed_any = False
        
        # Iterate all message contents
        for message in self.contents:
            if "content" in message and isinstance(message["content"], list):
                for i, content_item in enumerate(message["content"]):
                    # Find image_url content
                    if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                        image_url = content_item.get("image_url", {}).get("url", "")
                        
                        # Check if it is a base64 encoded image
                        if image_url.startswith("data:image/"):
                            # Parse format and base64 data
                            format_part = image_url.split(";")[0].split("/")[1]
                            base64_data = image_url.split(",")[1]
                            
                            try:
                                # Decode base64 data
                                image_data = base64.b64decode(base64_data)
                                img = Image.open(io.BytesIO(image_data))
                                
                                # Record original size
                                orig_width, orig_height = img.size
                                
                                # Resize to 50% of original
                                new_size = (int(orig_width * 0.5), int(orig_height * 0.5))
                                img = img.resize(new_size, Image.LANCZOS)
                                
                                # Compress image
                                output = io.BytesIO()
                                
                                # Ensure correct format when saving
                                save_format = "JPEG" if format_part.lower() in ["jpg", "jpeg"] else "PNG"
                                # Use 85% quality for JPEG
                                if save_format == "JPEG":
                                    img.save(output, format=save_format, quality=85, optimize=True)
                                else:
                                    img.save(output, format=save_format, optimize=True)
                                
                                # Convert to new base64
                                new_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
                                
                                # Calculate compression ratio
                                orig_size = len(base64_data)
                                new_size = len(new_base64)
                                compress_ratio = (1 - new_size/orig_size) * 100
                                
                                # Update content
                                new_url = f"data:image/{format_part};base64,{new_base64}"
                                message["content"][i]["image_url"]["url"] = new_url
                                
                                compressed_any = True
                                print(f"Compress image: {orig_width}x{orig_height} → {img.width}x{img.height} "
                                    f"(reduced by {compress_ratio:.1f}%)")
                            except Exception as e:
                                print(f"Error occurred while compressing image: {e}")
        
        return compressed_any