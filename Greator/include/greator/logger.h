// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "windows_customizations.h"
#include <iostream>

namespace greator {
#if defined(DISKANN_DLL)
extern std::basic_ostream<char> cout;
extern std::basic_ostream<char> cerr;
#else
DISKANN_DLLIMPORT extern std::basic_ostream<char> cout;
DISKANN_DLLIMPORT extern std::basic_ostream<char> cerr;
#endif

} // namespace diskann
