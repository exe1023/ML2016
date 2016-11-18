#!/bin/bash

KERAS_BACKEND=theano THEANO_FLAGS=device=gpu0 python3 self_train.py $1 $2
