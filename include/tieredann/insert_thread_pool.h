#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <string>
#include <atomic>

#include "diskann/index_factory.h"

namespace tieredann {

template <typename T, typename TagT = uint32_t>
class InsertThreadPool {
public:
    using TaskFn = std::function<void(std::unique_ptr<diskann::AbstractIndex>&, std::vector<TagT>, std::string, size_t)>;

    InsertThreadPool(size_t thread_count, TaskFn task_fn);
    ~InsertThreadPool();

    InsertThreadPool(const InsertThreadPool&) = delete;
    InsertThreadPool& operator=(const InsertThreadPool&) = delete;

    void submit(std::unique_ptr<diskann::AbstractIndex>& index,
                std::vector<TagT> to_be_inserted,
                const std::string& data_path,
                size_t dim);

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> stop;
    TaskFn task_function;
};

} 
