import random
import re
import numpy as np

def check_answer(args, item, ai_answer, discriminator=None):
    type_id = item.get('type_id')
    if type_id == 1:
        options = item.get('options')
        answer = item.get('answer')
        is_true = check_multi_choice_single(options, answer, ai_answer)

    elif type_id == 2:
        answer = item.get('answer')
        is_true = check_partial_open_question(answer, ai_answer)
    
    elif type_id == 3:
        answer = item.get('answer')
        is_true = check_partial_open_question(answer, ai_answer)

    elif type_id == 4:
        options = item.get('options')
        answer = item.get('answer')
        is_true = check_multi_choice_single(options, answer, ai_answer)
    
    elif type_id == 5:
        options = item.get('options')
        answer = item.get('answer')
        is_true = check_multi_choice_multi(options, answer, ai_answer)
    
    elif type_id == 6:
        options = item.get('options')
        answer = item.get('answer')
        is_true = check_multi_choice_single(options, answer, ai_answer)
    
    elif type_id == 7:
        answer = item.get('answer')
        is_true = check_partial_open_question(answer, ai_answer)
    
    elif type_id == 8:
        answer = item.get('answer')
        is_true = check_partial_open_question(answer, ai_answer)

    elif type_id == 9:
        options = item.get('options')
        answer = item.get('answer')
        is_true = check_multi_choice_multi(options, answer, ai_answer)

    elif type_id == 10:
        answer = item.get('answer')
        is_true = check_total_open_question(answer, ai_answer, discriminator)

    elif type_id == 11:
        answer = item.get('answer')
        is_true = check_box_answer(answer, ai_answer)

    return is_true


def check_multi_choice_single(options, answer, response):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in options.keys():  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in options.keys(): # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in options.keys(): # e.g., A: B: C: D:
            if f'{choice}: ' in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in options.keys(): # e.g., A\n B\n C\n D\n
            if f'{choice}\n' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in options.items():
            ans_str = str(ans)
            if ans_str.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(list(options.keys()))
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(options[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(answer, list):
        for ans in answer:
            if ans == pred_index:
                correct = True
                break
    else: # gold_i is a string
        if answer == pred_index:
            correct = True
    return correct

def check_multi_choice_multi(options, answer, response):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    candidates = []
    for choice in options.keys():  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)

    if len(candidates) == 0:
        for choice in options.keys(): # e.g., A B C D
            if f'{choice},' in response or f',{choice}' in response or f' {choice}' in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in options.keys(): # e.g., A: B: C: D:
            if f'{choice}: ' in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in options.keys(): # e.g., A\n B\n C\n D\n
            if f'{choice}\n' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in options.items():
            if ans.lower() in response.lower():
                candidates.append(index)

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = list(random.choice(list(options.keys())))
    else:
        pred_index = candidates.copy()

    # only they are exactly the same, we consider it as correct
    if isinstance(answer, list):
        for ans in answer:
            if ans not in pred_index:
                return False
            else:
                pred_index.remove(ans) # remove the correct answer from the pred_index
    else: # gold_i is a string
        if answer not in pred_index:
            return False
        else:
            pred_index.remove(answer)

    return len(pred_index) == 0 # if all the pred_index are correct, return True, otherwise False.

def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        # check if there's comma inside
        return False

def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    string = string.strip()
    is_number = check_is_number(string)
    if is_number:
        string = string.replace(',', '')
        num = float(string)
        if np.isinf(num) or np.isnan(num):
            return [string]
        string = round(num)
        return [string]
    else:
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]
        return [string]

def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_partial_open_question(answer, response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        response = response.strip().strip(".").lower()
        sub_responses = [resp.strip() for resp in re.split(r'\.\s(?=[A-Z])|\n|,', response)]
        return sub_responses

        
    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy() # keep the original string response
    for resp in key_responses:
        numbers = extract_numbers(resp)
        if len(numbers) > 0:
            pred_list.remove(resp) # remove the original string response
            pred_list.extend(numbers)

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    key_answers = get_key_subresponses(answer)

    ans_list = key_answers.copy() # keep the original string response
    for resp in key_answers:
        numbers = extract_numbers(resp)
        if len(numbers) > 0:
            ans_list.remove(resp)
            ans_list.extend(numbers)

    tmp_ans_list = []
    for i in range(len(ans_list)):
        tmp_ans_list.extend(normalize_str(ans_list[i]))
    
    norm_answers = tmp_ans_list.copy()
    
    for ans in norm_answers:
        if not ans in pred_list:
            return False

    return True # all the pred_list are correct.

def check_total_open_question(answer, response, discriminator):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    # content = content.strip("\n").strip(".").strip(" ")
    # 
    is_true = discriminator(answer, response)

    return is_true

def check_box_answer(answer, response, threshold=0.5):

    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    # content = content.strip("\n").strip(".").strip(" ")
    # 
    box = extract_bounding_box(str(response))
    if len(box) == 0:
        return False
    # Convert the string to a list of floats
    x_min1, y_min1, x_max1, y_max1 = box
    
    if x_min1 > x_max1 or y_min1 > y_max1:
        return 0
    x_min2, y_min2, x_max2, y_max2 = answer

    # Calculate intersection coordinates
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Calculate intersection area
    if x_max_inter > x_min_inter and y_max_inter > y_min_inter:
        area_inter = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    else:
        area_inter = 0

    # Calculate area of each box
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Calculate union area
    area_union = area1 + area2 - area_inter

    # Calculate IoU
    iou = area_inter / area_union if area_union != 0 else 0

    return iou > threshold # if iou > threshold, return 1, otherwise 0.

def extract_bounding_box(string):
    """
    Given a string returned by a model, use regular expressions to extract the bounding box prediction
    with support for floating point numbers. The string is expected to contain a sequence of four numbers
    (which represent xmin, ymin, xmax, ymax) formatted as a comma-separated list.
    
    Args:
    string (str): The string containing the bounding box coordinates in the expected format.
    
    Returns:
    list: A list of four floats [xmin, ymin, xmax, ymax] if the pattern is found, otherwise an empty list.
    """
    # Regex to find four floating point numbers, possibly surrounded by other text
    # The pattern allows optional leading sign, integer and fractional parts
    pattern = re.compile(r'\b([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\b')
    match = pattern.search(string)
    if match:
        # Extracts and converts all matched groups to floats
        return list(map(float, match.groups()))
    else:
        # Return an empty list if no matching pattern is found
        return []