#!/bin/bash

(cd C++\ code && make main)

mkdir -p output_files

#python3 main.py config_files/testing.ini

if [ $# -ge 1 ]
then
    if [ $1 = "-a" ]
    then
	for cur_file in config_files/*
	do
	    python3 main.py $cur_file
	done
    fi
else
    python3 main.py config_files/testing.ini
fi


