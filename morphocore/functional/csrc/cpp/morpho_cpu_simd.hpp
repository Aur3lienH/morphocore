#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// SIMD support of the CPU kernels.
//
// The kernels are written once against MorphoSimd<T, BYTES>, a native vector of BYTES bytes built on
// the GCC / Clang vector extensions, and are compiled twice:
//   - a generic version with 16 byte vectors, which only needs SSE2 and runs on any x86-64 CPU;
//   - an AVX2 version with 32 byte vectors, compiled with MORPHO_TARGET_AVX2.
// morpho_cpu_dispatch() picks one at runtime, so a wheel built for plain x86-64 still uses AVX2
// wherever it exists. Everything a kernel calls must be MORPHO_INLINE so that it is compiled for the
// instruction set of the entry point that inlines it.

#define MORPHO_INLINE inline __attribute__((always_inline))

#if defined(__x86_64__) || defined(__i386__)
    #define MORPHO_CPU_HAS_AVX2_PATH 1
    #define MORPHO_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
    #define MORPHO_CPU_HAS_AVX2_PATH 0
    #define MORPHO_TARGET_AVX2
#endif

/// @brief Vector width of the generic kernels (SSE2 on x86-64, NEON on arm64).
inline constexpr int MORPHO_SIMD_GENERIC_BYTES = 16;
/// @brief Vector width of the AVX2 kernels.
inline constexpr int MORPHO_SIMD_AVX2_BYTES = 32;

#if MORPHO_CPU_HAS_AVX2_PATH && defined(_WIN32)
#include <intrin.h>

/// @brief CPUID check of AVX2 and FMA, for Windows where clang-cl has no runtime behind
/// __builtin_cpu_supports. AVX also needs the OS to save the YMM registers (XGETBV bits 1 and 2).
__attribute__((target("xsave"))) inline bool morpho_cpuid_has_avx2_fma()
{
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7)
        return false;
    __cpuid(regs, 1);
    const bool fma = regs[2] & (1 << 12), osxsave = regs[2] & (1 << 27), avx = regs[2] & (1 << 28);
    if (!fma || !osxsave || !avx || (_xgetbv(0) & 6) != 6)
        return false;
    __cpuidex(regs, 7, 0);
    return regs[1] & (1 << 5);
}
#endif

/// @brief Whether the AVX2 kernels can run on this machine.
/// Setting MORPHOCORE_DISABLE_AVX2=1 forces the generic kernels, to compare both or to work around a problem.
inline bool morpho_cpu_has_avx2()
{
#if MORPHO_CPU_HAS_AVX2_PATH
#if defined(_WIN32)
    const bool supported = morpho_cpuid_has_avx2_fma();
#else
    const bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    static const bool available = supported && std::getenv("MORPHOCORE_DISABLE_AVX2") == nullptr;
    return available;
#else
    return false;
#endif
}

/// @brief Pick the AVX2 or the generic entry point of a kernel.
template <typename Fn>
Fn morpho_cpu_dispatch(Fn avx2, Fn generic)
{
    return morpho_cpu_has_avx2() ? avx2 : generic;
}

/// @brief One native vector of T and the matching comparison mask.
/// @tparam BYTES Vector width in bytes, one of MORPHO_SIMD_GENERIC_BYTES and MORPHO_SIMD_AVX2_BYTES.
template <typename T, int BYTES>
struct MorphoSimd
{
    /// Integer with the size of T: comparisons of two vec yield a mask of it, and it stores tap indices.
    using index_type = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
    static constexpr int64_t lanes = BYTES / sizeof(T);

    typedef T vec __attribute__((vector_size(BYTES)));
    typedef index_type mask __attribute__((vector_size(BYTES)));

    static MORPHO_INLINE vec load(const T* p)
    {
        vec v;
        std::memcpy(&v, p, sizeof(vec));
        return v;
    }

    static MORPHO_INLINE void store(T* p, vec v) { std::memcpy(p, &v, sizeof(vec)); }

    /// @brief Store the lanes of an index mask as int32 (a narrowing conversion when T is double).
    static MORPHO_INLINE void store_index(int32_t* p, mask m)
    {
        typedef int32_t narrow __attribute__((vector_size(lanes * sizeof(int32_t))));
        const narrow n = __builtin_convertvector(m, narrow);
        std::memcpy(p, &n, sizeof(narrow));
    }

    static MORPHO_INLINE vec splat(T x) { return vec{} + x; }

    static MORPHO_INLINE mask splat_index(index_type x) { return mask{} + x; }

    /// @brief Lane wise m ? a : b.
    template <typename V>
    static MORPHO_INLINE V select(mask m, V a, V b)
    {
        return m ? a : b;
    }

    /// @brief {0, 1, ..., lanes - 1}.
    static MORPHO_INLINE mask iota()
    {
        mask m;
        for (int64_t j = 0; j < lanes; j++)
        {
            m[j] = static_cast<index_type>(j);
        }
        return m;
    }

    /// @brief Lane wise exp(x), within 1 ulp of std::exp for T = float (2 for double).
    /// Cody-Waite reduction x = n ln2 + r with |r| <= ln2 / 2, then exp(r) by its Taylor series, whose
    /// truncation error is below half an ulp at that degree, and 2^n built in the exponent bits.
    /// Underflows to 0 like std::exp, keeps NaN; inputs are clamped below the overflow threshold.
    static MORPHO_INLINE vec exp(vec x)
    {
        constexpr bool single = sizeof(T) == 4;
        constexpr T max_input = single ? T(88.0) : T(709.0);
        constexpr T min_input = single ? T(-87.33654) : T(-708.39641);  // ln of the smallest normal number.
        constexpr T log2e = T(1.44269504088896340736);
        constexpr T ln2_hi = single ? T(0.693359375) : T(6.93145751953125e-1);
        constexpr T ln2_lo = single ? T(-2.12194440e-4) : T(1.42860682030941723212e-6);
        constexpr int degree = single ? 7 : 13;
        constexpr int mantissa_bits = single ? 23 : 52;
        constexpr index_type exponent_bias = single ? 127 : 1023;
        // Adding then subtracting 1.5 * 2^mantissa_bits rounds to the nearest integer.
        constexpr T round_magic = single ? T(12582912.0) : T(6755399441055744.0);

        const mask underflow = x < min_input;
        vec xc = select(x > max_input, splat(max_input), x);
        xc = select(underflow, splat(min_input), xc);

        const vec n = (xc * log2e + round_magic) - round_magic;
        const vec r = (xc - n * ln2_hi) - n * ln2_lo;

        // Horner scheme of sum r^k / k!, k = 0..degree.
        T inverse_factorial[degree + 1];
        inverse_factorial[0] = T(1);
        for (int k = 1; k <= degree; k++)
        {
            inverse_factorial[k] = inverse_factorial[k - 1] / T(k);
        }
        vec p = splat(inverse_factorial[degree]);
        for (int k = degree - 1; k >= 0; k--)
        {
            p = p * r + inverse_factorial[k];
        }

        const mask exponent = (__builtin_convertvector(n, mask) + exponent_bias) << mantissa_bits;
        vec scale;
        std::memcpy(&scale, &exponent, sizeof(vec));
        return select(underflow, vec{}, p * scale);
    }
};
