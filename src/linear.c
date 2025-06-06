#include "stdio.h"

int main() {
	int a = 5;
	int *ptr = &a;
	printf("Hello world!\n");
	printf("%ld\n", sizeof(ptr));
	return 0;
}
