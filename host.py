

import os
import json
import requests

url = 'http://localhost:5000/complete'



while True:
    input("Make changes to file ./test.py and click ENTER")
    with open("test.py", "r") as txt_file:
        context = txt_file.read()
    
    print(context)
    modelContext = context.replace("\n", "\\n")

    data = {'context': modelContext, "top_p": 0.9, "temp": 0.75}

    x = requests.post(url, json = data, headers = {"Content-Type": "application/json"})
    res = json.loads(x.text)

    print("Prediction: ")
    print(res["completion"].replace("\n", "\\n"))

    
