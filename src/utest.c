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

	Vector* v1 = vec_zero(3); vec_rand(0.0f, 1.0f, v1);
	Vector* v2 = vec_zero(3); vec_rand(0.0f, 1.0f, v2);
	Vector* v3 = vec_zero(3);
	Vector* v4 = vec_zero(3);
	Vector* v5 = vec_zero(3);
	Vector* v6 = vec_zero(3);
	Matrix* m1 = matrix_zero(3, 3); matrix_rand(0.0f, 1.0f, m1);
	Matrix* m2 = matrix_zero(3, 3);
	Matrix* m3 = matrix_zero(3, 3);
	Matrix* m4 = matrix_zero(3, 3);
	Matrix* m5 = matrix_zero(3, 3);matrix_iden(m5);
	Matrix* m6 = matrix_zero(3, 12);matrix_iden(m6);

	vec_add_ip(v1, v2, v3);
	vec_mul_ip(v1, v2, v4);
	matrix_vec_mul_ip(m1, v1, v5);
	matrix_vec_mul_offset_ip(m1, v1, v5, v6);
	matrix_transpose_ip(m1, m2);
	column_row_vec_mul_ip(v1, v2, m3);
	vec_matrix_hadamard_ip(v1, m3, m4);

	info("Base Vector:");
	vec_dump(v1);
	newline();
	vec_dump(v2);
	newline();
	info("Base Matrix:");
	matrix_dump_raw(m1);
	newline();
	info("Identity square Matrix:");
	matrix_dump_raw(m5);
	newline();
	info("Identity rect Matrix:");
	matrix_dump_raw(m6);
	newline();

	info("Vector addition:");
	vec_dump(v3);
	info("Vector multiplication:");
	vec_dump(v4);
	info("Matrix Vector multiplication:");
	vec_dump(v5);
	info("Matrix Vector multiplication with offset:");
	vec_dump(v6);
	info("Matrix transposition:");
	matrix_dump_raw(m2);
	info("Column row multiplication:");
	matrix_dump_raw(m3);
	info("Matrix hadamard product:");
	matrix_dump_raw(m4);

	vec_deallocate(v1);
	vec_deallocate(v2);
	vec_deallocate(v3);
	vec_deallocate(v4);
	vec_deallocate(v5);
	vec_deallocate(v6);
	matrix_deallocate(m1);
	matrix_deallocate(m2);
	matrix_deallocate(m3);
	matrix_deallocate(m4);
	matrix_deallocate(m5);
	matrix_deallocate(m6);
}
