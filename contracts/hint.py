
class hint:
    def __init__(self, text, value):
        self.text = text
        self.value = value

    text = ""
    value: 0.0
    
class hint_request: 
    def __init__(self, id, text, top_p, temp, tokens_length, num_completions, token):
        self.id = id
        self.text = text
        self.top_p = top_p
        self.temp = temp
        self.tokens_length = tokens_length
        self.num_completions = num_completions
        self.token = token
    
    id = ""
    text = ""
    top_p = 0.9
    temp = 0.75
    tokens_length = 32
    num_completions = 4
    token = ""
    
class hint_response:
    def __init__(self, hint_id, hints, duration = 0):
        self.hint_id = hint_id
        self.hints = list(map(lambda x: x.__dict__, hints))
        self.duration = duration
    
    hint_id = ""
    hints = None