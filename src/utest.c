#include <cpuid.h>

#include "vector.h"
#include "matrix.h"

#include "logger.h"

int main() {
	info("Vector Library Type: %s", VECTOR_LIB_TYPE);
	info("Matrix Library Type: %s", MATRIX_LIB_TYPE);

	unsigned int eax, ebx, ecx, edx;
	__cpuid(1, eax, ebx, ecx, edx);
	info("CPUID Leaf 1 EAX: 0b%08x", eax);
	info("CPUID Leaf 1 EBX: 0b%08x", ebx);
	info("CPUID Leaf 1 ECX: 0b%08x", ecx);
	info("CPUID Leaf 1 EDX: 0b%08x", edx);
	__cpuid(7, eax, ebx, ecx, edx);
	info("CPUID Leaf 7 EAX: 0b%08x", eax);
	info("CPUID Leaf 7 EBX: 0b%08x", ebx);
	info("CPUID Leaf 7 ECX: 0b%08x", ecx);
	info("CPUID Leaf 7 EDX: 0b%08x", edx);

	Vector* v1 = vec_rand(3, 0.0f, 1.0f);
	Vector* v2 = vec_rand(3, 0.0f, 1.0f);
	Vector* v3 = vec_zero(3);
	Vector* v4 = vec_zero(3);
	Vector* v5 = vec_zero(3);
	Matrix* m1 = matrix_rand(3, 3, 0.0f, 1.0f);

	vec_add_ip(v1, v2, v3);
	vec_mul_ip(v1, v2, v4);
	matrix_vec_mul_ip(m1, v1, v5);

	info("Base Vector:");
	vec_dump(v1);
	newline();
	vec_dump(v2);
	newline();
	info("Base Matrix:");
	matrix_dump(m1);
	newline();

	info("Vector addition:");
	vec_dump(v3);
	info("Vector multiplication:");
	vec_dump(v4);
	info("Matrix Vector multiplication:");
	vec_dump(v5);

	vec_deallocate(v1);
	vec_deallocate(v2);
	vec_deallocate(v3);
	vec_deallocate(v4);
	vec_deallocate(v5);
	matrix_deallocate(m1);
}
