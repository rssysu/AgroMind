import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from config import cfg

def get_answer(args, item, model):
    if args.model in ["random"]:
        return model.get_random_answer(item)
    type_id = item.get('type_id')
    if type_id == 1:
        question = item.get('question', item.get('questions'))
        image_path = item.get('image_path')
        image_path = os.path.join(args.image_dir, image_path)
        options = item.get('options')
        answer = multi_text_choice_single(question, image_path, options, model)
    
    elif type_id == 2:
        question = item.get('question', item.get('questions'))
        image_path = item.get('image_path')
        image_path = os.path.join(args.image_dir, image_path)
        answer = count_question(question, image_path, model)
    
    elif type_id == 3:
        question = item.get('question', item.get('questions'))
        image_path = item.get('image_path')
        image_path = os.path.join(args.image_dir, image_path)
        answer = judge_question(question, image_path, model)
    
    elif type_id == 4:
        question = item.get('question', item.get('questions'))
        options = item.get('options')
        for key, value in options.items():
            options[key] = os.path.join(args.image_dir, value)
        # concat_images = args.model in cfg.SINGLE.MODEL
        concat_images = True
        answer = multi_image_choice_single(question, options, model, concat_images=concat_images)

    elif type_id == 5:
        question = item.get('question', item.get('questions'))
        options = item.get('options')
        for key, value in options.items():
            options[key] = os.path.join(args.image_dir, value)
        # concat_images = args.model in cfg.SINGLE.MODEL
        concat_images = True
        answer = multi_image_choice_multi(question, options, model, concat_images=concat_images)
    
    elif type_id == 6:
        question = item.get('question', item.get('questions'))
        image_path = item.get('image_path')
        image_path = os.path.join(args.image_dir, image_path)
        options = item.get('options')
        answer = multi_text_choice_single(question, image_path, options, model)

    elif type_id == 7:
        question = item.get('question', item.get('questions'))
        image_path = item.get('image_path')
        image_path = os.path.join(args.image_dir, image_path)
        answer = partial_open_question(question, image_path, model)

    elif type_id == 8:
        question = item.get('question', item.get('questions'))
        image1 = item.get('image1')
        image2 = item.get('image2')
        image1_path = os.path.join(args.image_dir, image1)
        image2_path = os.path.join(args.image_dir, image2)
        # concat_images = args.model in cfg.SINGLE.MODEL
        concat_images = True
        answer = multi_image_count_question(question, image1_path, image2_path, model, concat_images=concat_images)

    elif type_id == 9:
        question = item.get('question', item.get('questions'))
        image_path = item.get('image_path')
        image_path = os.path.join(args.image_dir, image_path)
        options = item.get('options')
        answer = multi_text_choice_multi(question, image_path, options, model)

    elif type_id == 10:
        question = item.get('question', item.get('questions'))
        image_path = item.get('image_path')
        image_path = os.path.join(args.image_dir, image_path)
        answer = total_open_question(question, image_path, model)

    elif type_id == 11:
        question = item.get('question', item.get('questions'))
        image_path = item.get('image_path')
        image_path = os.path.join(args.image_dir, image_path)
        answer = get_box_question(question, image_path, model)
        
        
    print(f"Answer: {answer}")
    return answer

def multi_text_choice_single(question, image_path, options, model):
    prompt = f"""Please answer the following question by selecting the most appropriate option from the given choices.
    
Question: {question}

Options:
"""
    
    # Add all options
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    
    prompt += "\nPlease respond with only the letter of the correct option (e.g., A, B, C, or D), without any explanation. There is only one correct answer.\n"

    model.add_image(image_path)
    model.add_message(prompt)
    response = model.get_response()
    model.clear_contents()
    return response

def multi_text_choice_multi(question, image_path, options, model):
    prompt = f"""Please answer the following question by selecting all appropriate options from the given choices.
    
Question: {question}

Options:
"""
    
    # Add all options
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    
    prompt += "\nPlease respond with the letters of all correct options (e.g., A or A,B or A,C,D) without any explanation. If there are multiple correct answers, separate them with commas."

    model.add_image(image_path)
    model.add_message(prompt)
    response = model.get_response()
    model.clear_contents()
    return response

def count_question(question, image_path, model):
    prompt = f"""Please answer the following question based on the provided image.

Question: {question}
"""
    prompt += "\nPlease respond with only a number as your answer, without any explanation.\n"

    model.add_image(image_path)
    model.add_message(prompt)
    response = model.get_response()
    model.clear_contents()
    return response



def judge_question(question, image_path, model):
    prompt = f"""Please answer the following question based on the provided image.

Question: {question}
"""
    prompt += "\nPlease respond with only 'yes' or 'no' as your answer, without any explanation.\n"

    model.add_image(image_path)
    model.add_message(prompt)
    response = model.get_response()
    model.clear_contents()
    return response

def multi_image_choice_single(question, options, model, concat_images=False):
    if not concat_images:
        # Add each image sequentially
        for key, image_path in options.items():
            model.add_image(image_path)
        
        # Add prompt
        prompt = f"""Based on the following question, please select the most appropriate image.

Question: {question}

I have shown you multiple images in the following order:
- The first image is option A
- The second image is option B
- The third image is option C
- The fourth image is option D

Please respond with only the letter of the correct option (A, B, C, or D) without any explanation.
"""
        model.add_message(prompt)
    else:
        # Concatenate images using matplotlib
        
        
        # Create concat_images directory if it doesn't exist
        os.makedirs("./concat_images", exist_ok=True)
        
        # Create path for concatenated image
        concat_image_path = os.path.join("./concat_images", "concat_image_temp.jpg")
        
        # Create a 2x2 grid for the images
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        # Flatten the axes array for easy indexing
        axs = axs.flatten()
        
        # Load and display each image
        for i, (key, image_path) in enumerate(options.items()):
            if i < 4:  # Ensure we only process up to 4 images
                img = plt.imread(image_path)
                axs[i].imshow(img)
                axs[i].axis('off')  # Turn off axis
                axs[i].set_title(f"Option {key}", fontsize=14)
        
        # Add spacing between subplots
        plt.tight_layout(pad=3.0)
        
        # Save the concatenated image
        plt.savefig(concat_image_path)
        plt.close()
        
        # Add concatenated image
        model.add_image(concat_image_path)
        
        # Add prompt
        prompt = f"""Based on the following question, please select the most appropriate image.

Question: {question}

I have shown you a grid of four images:
- Top-left is option A
- Top-right is option B
- Bottom-left is option C
- Bottom-right is option D

Please respond with only the letter of the correct option (A, B, C, or D) without any explanation.
"""
        model.add_message(prompt)
    
    # Get and return response
    response = model.get_response()
    model.clear_contents()
    return response

def multi_image_choice_multi(question, options, model, concat_images=False):
    if not concat_images:
        # Add each image sequentially
        for key, image_path in options.items():
            model.add_image(image_path)
        
        # Add prompt
        prompt = f"""Based on the following question, please select all appropriate images that match the criteria.

Question: {question}

I have shown you multiple images in the following order:
- The first image is option A
- The second image is option B
- The third image is option C
- The fourth image is option D

Please respond with the letters of all correct options (e.g., A or A,B or A,C,D) without any explanation. If there are multiple correct answers, separate them with commas.
"""
        model.add_message(prompt)
    else:
        # Concatenate images using matplotlib
        
        
        # Create concat_images directory if it doesn't exist
        os.makedirs("./concat_images", exist_ok=True)
        
        # Create path for concatenated image
        concat_image_path = os.path.join("./concat_images", "concat_image_temp.jpg")
        
        # Create a 2x2 grid for the images
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        # Flatten the axes array for easy indexing
        axs = axs.flatten()
        
        # Load and display each image
        for i, (key, image_path) in enumerate(options.items()):
            if i < 4:  # Ensure we only process up to 4 images
                img = plt.imread(image_path)
                axs[i].imshow(img)
                axs[i].axis('off')  # Turn off axis
                axs[i].set_title(f"Option {key}", fontsize=14)
        
        # Add spacing between subplots
        plt.tight_layout(pad=3.0)
        
        # Save the concatenated image
        plt.savefig(concat_image_path)
        plt.close()
        
        # Add concatenated image
        model.add_image(concat_image_path)
        
        # add prompt
        prompt = f"""Based on the following question, please select all appropriate images that match the criteria.

Question: {question}

I have shown you a grid of four images:
- Top-left is option A
- Top-right is option B
- Bottom-left is option C
- Bottom-right is option D

Please respond with the letters of all correct options (e.g., A or A,B or A,C,D) without any explanation. If there are multiple correct answers, separate them with commas.
"""
        model.add_message(prompt)
    
    # Get and return response
    response = model.get_response()
    model.clear_contents()
    return response

def partial_open_question(question, image_path, model):
    prompt = f"""Please answer the following question based on the provided image.

Question: {question}
"""
    prompt += "\nIf the question has specific formatting requirements, please follow them. Otherwise, answer the question using a single word or phrase.\n"

    model.add_image(image_path)
    model.add_message(prompt)
    response = model.get_response()
    model.clear_contents()
    return response

def multi_image_count_question(question, image1_path, image2_path, model, concat_images=False):
    if not concat_images:
        # Add each image sequentially
        model.add_image(image1_path)
        model.add_image(image2_path)
        
        # Add prompt
        prompt = f"""Please answer the following question based on the provided images.
    
Question: {question}
"""
        prompt += "\nPlease respond with only a number as your answer, without any explanation.\n"

        model.add_message(prompt)
    
    else:
        # Concatenate images using matplotlib
        
        
        # Create concat_images directory if it doesn't exist
        os.makedirs("./concat_images", exist_ok=True)
        
        # Create path for concatenated image
        concat_image_path = os.path.join("./concat_images", "concat_image_temp.jpg")
        
        # Create a 1x2 grid for the images
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Load and display each image
        for i, image_path in enumerate([image1_path, image2_path]):
            img = plt.imread(image_path)
            axs[i].imshow(img)
            axs[i].axis('off')  # Turn off axis
            axs[i].set_title(f"Image {i+1}", fontsize=14)
        
        # Add spacing between subplots
        plt.tight_layout(pad=3.0)

        # Save the concatenated image
        plt.savefig(concat_image_path)
        plt.close()
        # Add concatenated image
        model.add_image(concat_image_path)
        # Add prompt
        prompt = f"""Please answer the following question based on the provided images.

Question: {question}
"""
        prompt += "\nPlease respond with only a number as your answer, without any explanation.\n"

        model.add_message(prompt)
    
    response = model.get_response()
    model.clear_contents()
    return response

def total_open_question(question, image_path, model):
    prompt = f"""Please answer the following question based on the provided image.

Question: {question}
"""
    prompt += "\nPlease respond with a concise and clear answer, avoiding unnecessary details or explanations.\n"

    model.add_image(image_path)
    model.add_message(prompt)
    response = model.get_response()
    model.clear_contents()
    return response

def get_box_question(question, image_path, model):
    prompt = f"""Please answer the following question based on the provided image.

Question: {question}
"""
    prompt += "\nRespond with only four normalized floating-point values representing the bounding box coordinates: xmin,ymin,xmax,ymax.\n"
    prompt += "All values must be between 0 and 1, where:\n"
    prompt += "- (0,0) is the top-left corner of the image\n"
    prompt += "- x-coordinates run horizontally from left(0) to right(1)\n"
    prompt += "- y-coordinates run vertically from top(0) to bottom(1)\n"
    prompt += "Provide only these four comma-separated values without brackets, explanations, or any other text.\n"

    model.add_image(image_path)
    model.add_message(prompt)
    response = model.get_response()
    model.clear_contents()
    return response