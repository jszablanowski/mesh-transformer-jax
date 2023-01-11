import argparse
import json
import threading
import time
import jax
import numpy as np
import optax
import asyncio
import socketio
import eventlet
import json
import transformers


from eventlet.queue import LightQueue, Empty
from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
from smart_open import open
from mesh_transformer.util import clip_by_global_norm
from contracts.hint import hint_response, hint_request, hint


with open("config.json") as json_data_file:
    config = json.load(json_data_file)

sio = socketio.Server(async_mode='eventlet')
app = socketio.WSGIApp(sio)

ckpt_step = 39637
requests_queue = LightQueue()

@sio.event
def connect(sid, environ):
    print('connect ', sid)

    # TODO
    # check api key
    # if invalid api key, close websocket connection
    # sio.disconnect(sid)
    # if valid, continue

@sio.event
def get_completions(sid, packed_data):
    data = hint_request(**(json.loads(packed_data)))
    print("Received:")
    print(data.text)
    if requests_queue.qsize() > 100:
        return {"error": "queue full, try again later"}
    
    response_queue = LightQueue()
    start = time.time()

    for _ in range(data.num_completions):
        requests_queue.put(({
                                "context": data.text,
                                "top_p": data.top_p,
                                "temp": data.temp,
                                "tokens_length": data.tokens_length
                            }, response_queue))

    extracted_hints = []
    for _ in range(data.num_completions):
        model_response = response_queue.get()
        extracted_hints.append(hint(model_response["text"], model_response["probability"]))

    response = hint_response(data.id, extracted_hints, float(time.time() - start) * 1000)

    print("Response:")
    for h in extracted_hints:
        print(h.text)
        print("\nProbability: " + str(h.value))
        print("-------")
    
    sio.emit("receive_completions", json.dumps(response.__dict__))

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/minipilot.json", help="Config file location")

    args = parser.parse_args()
    return args

def server():
    eventlet.wsgi.server(eventlet.listen((config["address"], config["port"])), app)

def predictor(params):
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    seq = params["seq"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    print(f"using checkpoint {ckpt_step}")

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica * 8
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

        while True:
            all_ctx = []
            all_top_p = []
            all_temp = []
            gen_length = 32
            all_q = []
            while len(all_ctx) < total_batch:
                try:
                    o, q = requests_queue.get(block=False)
                    all_ctx.append(o["context"])
                    all_top_p.append(o["top_p"])
                    all_temp.append(o["temp"])
                    gen_length = o["tokens_length"]
                    all_q.append(q)
                except Empty:
                    if len(all_ctx):
                        break
                    else:
                        eventlet.sleep(0.01)

            start = time.time()
            while len(all_ctx) < total_batch:
                all_ctx.append("whatever")
                all_top_p.append(1)
                all_temp.append(1)

            all_tokenized = []
            all_length = []
            for ctx in all_ctx:
                padded_tokens = np.zeros(seq).astype(np.uint32)
                length = 0

                try:
                    tokens = tokenizer.encode(ctx)
                    provided_ctx = len(tokens)
                    pad_amount = seq - provided_ctx

                    pad_amount = max(pad_amount, 0)

                    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)[-seq:]
                    length = len(tokens)
                except:
                    print("oops exception")

                all_tokenized.append(padded_tokens)
                all_length.append(length)
            print(f"tokenizer encode done in {time.time() - start:06}s")
            
            start2 = time.time()
            output = network.generate(np.array(all_tokenized),
                                      np.array(all_length),
                                      gen_length,
                                      {
                                          "top_p": np.array(all_top_p),
                                          "temp": np.array(all_temp)
                                      },
                                      return_logits=True)
            print(f"inference done in {time.time() - start2:06}s")
            
            start3 = time.time()
            for o, l, q in zip(output[1][0][:, :, 0], output[1][2][:, :, 0], all_q):
                probability = 1.0
                for token, logits in zip(o, l):
                    # TODO: check if cpu softmax is faster
                    probability *= jax.nn.softmax(logits)[token]
                
                q.put({
                    "text": tokenizer.decode(o),
                    "probability": float(probability)
                })


            print(f"tokenizer decode done in {time.time() - start3:06}s")

            print(f"all completion done in {time.time() - start:06}s")

if __name__ == "__main__":
    pool = eventlet.GreenPool()

    args = parse_args()
    params = json.load(open(args.config))

    secret = config["secret"]
    pool.spawn(predictor, params)

    eventlet.wsgi.server(eventlet.listen((config["address"], config["port"])), app)