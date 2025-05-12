import random

class RandomClient:
    def __init__(self):
        self.type_func_map = {
            1: self._type_id_1,
            2: self._type_id_2,
            3: self._type_id_3,
            4: self._type_id_4,
            5: self._type_id_5,
            6: self._type_id_6,
            8: self._type_id_8,
            9: self._type_id_9,
            11: self._type_id_11,
        }

    def get_random_answer(self, item):
        type_id = item.get('type_id')
        func = self.type_func_map.get(type_id)
        if func:
            answer = func(item)
            return str(answer)
        else:
            return None

    def _type_id_1(self, item):
        options = item.get("options")
        return random.choice(list(options.keys()))
    
    def _type_id_2(self, item):
        return random.randint(0, 100)
    
    def _type_id_3(self, item):
        return random.choice(["Yes", "No"])
    
    def _type_id_4(self, item):
        options = item.get("options")
        return random.choice(list(options.keys()))
    
    def _type_id_5(self, item):
        import re
        options = list(item.get("options").keys())
        question = item.get("question", "")
        # Find the number after "which"
        match = re.search(r'which\s+(\d+)', question, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            num = min(num, len(options))
            return random.sample(options, num)
        else:
            return [random.choice(options)]
        
    def _type_id_6(self, item):
        options = item.get("options")
        return random.choice(list(options.keys()))
    
    def _type_id_8(self, item):
        return random.randint(0, 10)
    
    def _type_id_9(self, item):
        import re
        options = list(item.get("options").keys())
        question = item.get("question", "")
        # Find the number after "which"
        match = re.search(r'which\s+(\d+)', question, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            num = min(num, len(options))
            return random.sample(options, num)
        else:
            return [random.choice(options)]
        
    def _type_id_11(self, item):
        # Randomly generate a valid box, make sure x_min < x_max, y_min < y_max, all values in 0-1
        x_min = random.uniform(0, 0.8)
        y_min = random.uniform(0, 0.8)
        x_max = random.uniform(x_min + 0.01, 1.0)
        y_max = random.uniform(y_min + 0.01, 1.0)
        return [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)]