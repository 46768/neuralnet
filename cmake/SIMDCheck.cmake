include(CheckCSourceRuns)

# Get AVX2 availability
check_c_source_runs("
#include <cpuid.h>
int main() {
unsigned int eax,ebx,ecx,edx;
__cpuid_count(7,0,eax,ebx,ecx,edx);
return (ebx & (1 << 5)) == 0;
}" haveAVX2
)
# Get AVX availability
check_c_source_runs("
#include <cpuid.h>
int main() {
unsigned int eax,ebx,ecx,edx;
__cpuid_count(1,0,eax,ebx,ecx,edx);
return (edx & (1 << 28)) == 0;
}" haveAVX
)
