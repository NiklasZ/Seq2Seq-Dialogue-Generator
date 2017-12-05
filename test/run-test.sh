#!/bin/bash

../data/write_tuples.py "$1"
var="$1"
replace="_Tuples.txt"
../data/data.py ${var//.txt/$replace} 'test' --metadata metadata.pkl --crop &> /dev/null
python test.py 2> /dev/null
