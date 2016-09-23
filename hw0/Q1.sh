#!/bin/bash

n=$(($1+1))
ans=`awk -v awk_var="$n" '{print $awk_var}' $2 | sort -n | tr '\n' ','`

echo ${ans%?} > ans1.txt

