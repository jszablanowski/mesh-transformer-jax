
class hint:
    def __init__(self, text, value):
        self.text = text
        self.value = value

    text = ""
    value: 0.0
    
class hint_request: 
    def __init__(self, id, text):
        self.id = id
        self.text = text
    
    id = ""
    text = ""
    
class hint_response:
    def __init__(self, id, hints):
        self.id = id
        self.hints = list(map(lambda x: x.__dict__, hints))
    
    id = ""
    hints = None