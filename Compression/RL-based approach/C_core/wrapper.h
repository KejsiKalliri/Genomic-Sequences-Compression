#pragma once

#include <stddef.h>

#ifdef __cplusplus  //kjo pjesa siguron qe nqs wrapperi kompilohet me C++, emrat e funksioneve te mos ndryshohen nga compileri. Kjo është praktikë standarde për kompatibilitet C/C++.
extern "C" {
#endif

unsigned char* compress_block_bz2_c(  // ky eshte funksioni kryesore i kompresimit ne C
    const unsigned char* input,      // ky funx merr nje block ADN-je si unsigned char* input
    unsigned int input_len,         // gjatesine e tij
    int level,                     // nivelin e kompresimit
    unsigned int* output_len      // dhe kthen nje buffer te kompresuar
);

unsigned char* decompress_block_bz2_c(
    const unsigned char* input,
    unsigned int input_len,
    unsigned int expected_output_len,
    unsigned int* output_len
);

void free_bz2_buffer(void* p);

#ifdef __cplusplus
}
#endif