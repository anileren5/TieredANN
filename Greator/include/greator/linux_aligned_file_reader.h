// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifndef _WINDOWS

#include "aligned_file_reader.h"


namespace greator {

class LinuxAlignedFileReader : public AlignedFileReader {
private:
  uint64_t file_sz;
  FileHandle file_desc;
  io_context_t bad_ctx = (io_context_t)-1;

public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  IOContext &get_ctx();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false);

  void sequential_write(AlignedRead &write_req, IOContext &ctx);
  void write(std::vector<AlignedRead> &write_reqs, IOContext &ctx);
};

#endif
}