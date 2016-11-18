#!/bin/bash

KERAS_BACKEND=theano THEANO_FLAGS=device=gpu0 python3 aetrain.py $1 $2
