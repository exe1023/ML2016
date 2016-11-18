#!/bin/bash

KERAS_BACKEND=theano THEANO_FLAGS=device=gpu0 python3 test.py $2 $1 prediction
cat prediction | tr -d ' ' > $3
rm prediction
