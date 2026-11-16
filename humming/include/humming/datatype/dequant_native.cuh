#pragma once

#include <humming/datatype/dequant_prepare.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


// CUDA 11.8, 12.7 and 13.2 are the first toolkits with PTX ISA 7.8, 8.6 and 9.2.
constexpr bool kNativeDequantPtx78Available =
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8))
    true;
#else
    false;
#endif


constexpr bool kNativeDequantLowBitFamily =
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
    true;
#else
    false;
#endif


constexpr bool kNativeDequantPtx86Available =
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 7))
    kNativeDequantLowBitFamily;
#else
    false;
#endif


constexpr bool kNativeDequantPtx92Available =
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
    kNativeDequantLowBitFamily;
#else
    false;
#endif


template <class SourceType, class TargetType>
constexpr bool kNativeDequantSupported =
    std::is_same<TargetType, Float16>::value
        ? (kNativeDequantPtx78Available &&
               (std::is_same<SourceType, Float8E4M3>::value || std::is_same<SourceType, Float8E5M2>::value) ||
           kNativeDequantPtx86Available &&
               (std::is_same<SourceType, Float4E2M1>::value ||
                std::is_same<SourceType, Float6E3M2>::value ||
                std::is_same<SourceType, Float6E2M3>::value))
        : std::is_same<TargetType, BFloat16>::value &&
              kNativeDequantPtx92Available &&
              (std::is_same<SourceType, Float4E2M1>::value ||
               std::is_same<SourceType, Float6E3M2>::value ||
               std::is_same<SourceType, Float6E2M3>::value || std::is_same<SourceType, Float8E4M3>::value ||
               std::is_same<SourceType, Float8E5M2>::value);


// Odd-code E2M1 weights are stored in the native layout on every toolkit; dequant_native converts them with the
// hardware E2M1 instruction when the PTX has it and in software otherwise.
template <class SourceType, class TargetType>
constexpr bool kUseNativeWeightDequant =
    kNativeDequantSupported<SourceType, TargetType> ||
    std::is_same<SourceType, Float3E2M1Odd>::value &&
        (std::is_same<TargetType, Float16>::value || std::is_same<TargetType, BFloat16>::value);


template <class SourceType, class TargetType>
CUDA_INLINE uint32_t dequant_native_x2(uint16_t packed) {
  static_assert(kNativeDequantSupported<SourceType, TargetType>);
  uint32_t result;

  if constexpr (std::is_same<SourceType, Float4E2M1>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile(
          "{ .reg .b8 a, b; mov.b16 {a, b}, %1; cvt.rn.f16x2.e2m1x2 %0, a; }"
          : "=r"(result) : "h"(packed));
    } else {
      asm volatile(
          "{ .reg .b8 a, b; mov.b16 {a, b}, %1; cvt.rn.bf16x2.e2m1x2 %0, a; }"
          : "=r"(result) : "h"(packed));
    }
  } else if constexpr (std::is_same<SourceType, Float6E3M2>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile("cvt.rn.f16x2.e3m2x2 %0, %1;" : "=r"(result) : "h"(packed));
    } else {
      asm volatile("cvt.rn.bf16x2.e3m2x2 %0, %1;" : "=r"(result) : "h"(packed));
    }
  } else if constexpr (std::is_same<SourceType, Float6E2M3>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile("cvt.rn.f16x2.e2m3x2 %0, %1;" : "=r"(result) : "h"(packed));
    } else {
      asm volatile("cvt.rn.bf16x2.e2m3x2 %0, %1;" : "=r"(result) : "h"(packed));
    }
  } else if constexpr (std::is_same<SourceType, Float8E4M3>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(packed));
    } else {
      asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(packed));
    }
  } else if constexpr (std::is_same<SourceType, Float8E5M2>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(result) : "h"(packed));
    } else {
      asm volatile("cvt.rn.bf16x2.e5m2x2 %0, %1;" : "=r"(result) : "h"(packed));
    }
  }
  return result;
}


template <class TargetType>
CUDA_INLINE void dequant_e2m1_native_x8(uint32_t packed, uint32_t *dst) {
  static_assert(kNativeDequantSupported<Float4E2M1, TargetType>);
  if constexpr (std::is_same<TargetType, Float16>::value) {
    asm volatile(
        "{ .reg .b8 a, b, c, d; mov.b32 {a, b, c, d}, %4; "
        "cvt.rn.f16x2.e2m1x2 %0, a; cvt.rn.f16x2.e2m1x2 %1, b; "
        "cvt.rn.f16x2.e2m1x2 %2, c; cvt.rn.f16x2.e2m1x2 %3, d; }"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(packed));
  } else {
    asm volatile(
        "{ .reg .b8 a, b, c, d; mov.b32 {a, b, c, d}, %4; "
        "cvt.rn.bf16x2.e2m1x2 %0, a; cvt.rn.bf16x2.e2m1x2 %1, b; "
        "cvt.rn.bf16x2.e2m1x2 %2, c; cvt.rn.bf16x2.e2m1x2 %3, d; }"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(packed));
  }
}


template <class SourceType, class TargetType>
CUDA_INLINE void dequant_native_x8(uint32_t packed0, uint32_t packed1, uint32_t *dst) {
  static_assert(kNativeDequantSupported<SourceType, TargetType>);
  uint16_t packed[4];
  asm volatile(
      "mov.b32 {%0, %1}, %4; mov.b32 {%2, %3}, %5;"
      : "=h"(packed[0]), "=h"(packed[1]), "=h"(packed[2]), "=h"(packed[3])
      : "r"(packed0), "r"(packed1));

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    dst[i] = dequant_native_x2<SourceType, TargetType>(packed[i]);
  }
}


// Three words of Float3E2M1Odd codes, packed as humming_pack_weight<3> packs 3-bit values, hold 32 weights; word
// `index` (0..3) of the four FP4 words they expand to. Each code sits in a nibble whose low bit is the
// always-set mantissa, so expanding is masks and shifts plus OR-ing that bit back.
CUDA_INLINE uint32_t expand_e2m1_odd_word(const uint32_t *src, uint32_t index) {
  const uint32_t *group = src + index / 4 * 3;
#ifdef HUMMING_DEBUG_CHEAP_ODD_EXPAND
  return group[index % 3] | 0x11111111u;
#endif
  switch (index % 4) {
    case 0: return ((group[0] & 0x77777777u) << 1) | 0x11111111u;
    case 1: return (group[0] & 0x88888888u) | ((group[1] & 0x33333333u) << 1) | 0x11111111u;
    case 2: return (group[1] & 0xCCCCCCCCu) | ((group[2] & 0x11111111u) << 1) | 0x11111111u;
    default: return (group[2] & 0xEEEEEEEEu) | 0x11111111u;
  }
}


// Software equivalent of dequant_e2m1_native_x8 for words whose nibbles all have mantissa bit 1 ({±0.5, ±1.5,
// ±3, ±6}): each nibble's exponent e and sign s give the 16-bit value directly. Byte i fills dst[i], low nibble
// in the low half, as the hardware conversion does.
template <class TargetType>
CUDA_INLINE void dequant_e2m1_odd_soft_x8(uint32_t packed, uint32_t *dst) {
  constexpr bool kIsHalf = std::is_same<TargetType, Float16>::value;
  constexpr uint32_t kBase = kIsHalf ? 0x3800u : 0x3F00u;      // 0.5
  constexpr uint32_t kExpShift = kIsHalf ? 10 : 7;
  constexpr uint32_t kHalfMantissa = kIsHalf ? 0x200u : 0x40u;  // the .5 of 1.5 x 2^k
  auto convert = [&](uint32_t nibble) {
    uint32_t e = (nibble >> 1) & 3u;
    uint32_t bits = kBase + (e << kExpShift) + (e != 0u ? kHalfMantissa : 0u);
    return bits | ((nibble & 8u) << 12);
  };
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t byte = (packed >> (8 * i)) & 0xFFu;
    dst[i] = convert(byte & 0xFu) | (convert(byte >> 4) << 16);
  }
}


template <class SourceType, class TargetType, uint32_t kNumWarpShapeNSplits = 1>
CUDA_INLINE void dequant_native(const uint32_t *src, uint32_t *dst, uint32_t index) {
  uint32_t packed[2];
  if constexpr (std::is_same<SourceType, Float4E2M1>::value) {
    dequant_e2m1_native_x8<TargetType>(src[index], dst);
  } else if constexpr (std::is_same<SourceType, Float3E2M1Odd>::value) {
    // Odd-width weights are not split per warp: with split warps each thread holds the whole 32-value group,
    // and odd warps take its second half (as dequant_b3567 does).
    if (kNumWarpShapeNSplits == 2 && threadIdx.x / 32 % 2 == 1) index += 2;
#ifndef HUMMING_DEBUG_SOFT_ODD_E2M1
    if constexpr (kNativeDequantSupported<Float4E2M1, TargetType>) {
      dequant_e2m1_native_x8<TargetType>(expand_e2m1_odd_word(src, index), dst);
    } else
#endif
    {
      dequant_e2m1_odd_soft_x8<TargetType>(expand_e2m1_odd_word(src, index), dst);
    }
  } else {
    if constexpr (SourceType::kBits == 6) {
      packed[0] = get_quanted_value_group<6, false>(src, index * 2);
      packed[1] = get_quanted_value_group<6, false>(src, index * 2 + 1);
      packed[0] = (packed[0] & 0xFCFCFCFCu) >> 2;
      packed[1] = (packed[1] & 0xFCFCFCFCu) >> 2;
    } else {
      packed[0] = src[index * 2];
      packed[1] = src[index * 2 + 1];
    }
    dequant_native_x8<SourceType, TargetType>(packed[0], packed[1], dst);
  }
}
