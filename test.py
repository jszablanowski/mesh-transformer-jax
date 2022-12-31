from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./minipilot_slim_hf", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = (
    "def factorial(n):\n"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]


print(gen_text)
print("-----------------------------")

gen_text = tokenizer.batch_decode(gen_tokens)[0]


print(gen_text)
print("-----------------------------")


gen_text = tokenizer.batch_decode(gen_tokens)[0]


print(gen_text)
print("-----------------------------")
gen_text = tokenizer.batch_decode(gen_tokens)[0]


print(gen_text)
print("-----------------------------")

gen_text = tokenizer.batch_decode(gen_tokens)[0]


print(gen_text)
print("-----------------------------")
