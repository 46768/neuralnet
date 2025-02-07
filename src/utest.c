#include <cpuid.h>

#include "vector.h"
#include "matrix.h"

#include "logger.h"

int main() {
	info("Vector Library Type: %s", VECTOR_LIB_TYPE);
	info("Matrix Library Type: %s", MATRIX_LIB_TYPE);

	unsigned int eax, ebx, ecx, edx;
	__cpuid(1, eax, ebx, ecx, edx);
	info("CPUID Leaf 1 EAX: 0b%08x\n", eax);
	info("CPUID Leaf 1 EBX: 0b%08x\n", ebx);
	info("CPUID Leaf 1 ECX: 0b%08x\n", ecx);
	info("CPUID Leaf 1 EDX: 0b%08x\n", edx);
	__cpuid(7, eax, ebx, ecx, edx);
	info("CPUID Leaf 7 EAX: 0b%08x\n", eax);
	info("CPUID Leaf 7 EBX: 0b%08x\n", ebx);
	info("CPUID Leaf 7 ECX: 0b%08x\n", ecx);
	info("CPUID Leaf 7 EDX: 0b%08x\n", edx);
}
