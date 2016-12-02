#!/bin/bash

wget https://www.csie.ntu.edu.tw/~b03902071/w2vmodel -O w2vmodel
python3 train.py $1 $2 w2vmodel