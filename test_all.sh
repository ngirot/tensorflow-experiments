#!/bin/sh

for i in $(ls *.py) ; do
	echo '>>> ' $i 
	for j in {0..5}; do
		python $i 2> /dev/null | grep "0\."
	done
done
