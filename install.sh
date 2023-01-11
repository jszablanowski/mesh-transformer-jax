#!/usr/bin/env bash

pip install -r requirements.txt

./scripts/init_ray.sh

pip install jaxlib==0.1.67