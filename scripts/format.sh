#!/usr/bin/bash
for i in $(find ./lib | grep -E ".*(\.h|\.c|\.hpp|\.cpp)$")
do
	echo "Formatting file $i"
	clang-format -i $i
done
for i in $(find ./src | grep -E ".*(\.h|\.c|\.hpp|\.cpp)$")
do
	echo "Formatting file $i"
	clang-format -i $i
done
