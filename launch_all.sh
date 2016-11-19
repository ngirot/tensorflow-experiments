#!/bin/sh

for i in $(ls *.py) ; do
	echo '>>> ' $i
	sum_iteration=0
	for j in {1..5}; do
		score=$(python $i 2> /dev/null | grep "0\.")
		sum_iteration=$(echo $sum_iteration + $score | bc)
	done

	bc -l <<< "scale=4; $sum_iteration/5"
done
