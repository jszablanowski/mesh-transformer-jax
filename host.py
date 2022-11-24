

import os

import requests

url = 'http://localhost:5000/complete'



while True:
    input("Make changes to file ./test.py and click ENTER")
    with open("test.py", "r") as txt_file:
        context = txt_file.read()
    
    print(context)
    modelContext = context.replace("\n", "\\n")

    data = {'context': 'aaa', "top_p": 0.9, "temp": 0.75}

    x = requests.post(url, json = data, headers = {"Content-Type": "application/json"})
    

    print("Prediction: ")
    print(x.text)

    
