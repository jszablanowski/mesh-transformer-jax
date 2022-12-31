from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
from jax import pmap

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert "TPU" in device_type, "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"


model = AutoModelForCausalLM.from_pretrained("./minipilot_slim_hf", dtype=jnp.float16, revision="f16")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = (
    "def factorial(n):\n"
)

input_ids = tokenizer(prompt, return_tensors="jax").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
