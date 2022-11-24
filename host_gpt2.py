import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation", model="./200k"
)


while True:
    input("Make changes to file ./test.py and click ENTER")
    with open("test.py", "r") as txt_file:
        context = txt_file.read()
    
    print(context)
    modelContext = context.replace("\n", "\\n")


    x = pipe(modelContext, num_return_sequences=1)[0]["generated_text"]
    

    print("Prediction: ")
    print(x)

