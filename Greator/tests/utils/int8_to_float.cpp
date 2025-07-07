// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    greator::cout << argv[0] << " input_int8_bin output_float_bin" << std::endl;
    exit(-1);
  }

  int8_t *input;
  size_t npts, nd;
  greator::load_bin<int8_t>(argv[1], input, npts, nd);
  float *output = new float[npts * nd];
  greator::convert_types<int8_t, float>(input, output, npts, nd);
  greator::save_bin<float>(argv[2], output, npts, nd);
  delete[] output;
  delete[] input;
}
