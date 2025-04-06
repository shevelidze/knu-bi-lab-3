#! /bin/bash

head -n 1 data.csv > light-data.csv
shuf -n 100000 data.csv >> light-data.csv