#!/usr/bin/bash
for i in $(find . -not -path "./build/*" | grep -E ".*(\.h|\.c)$")
do
	echo "Formatting file $i"
	clang-format -i $i
done
