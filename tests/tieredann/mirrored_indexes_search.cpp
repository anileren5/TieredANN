// System headers
#include <iomanip>
#include <omp.h>
#include <boost/program_options.hpp>
#include <future>

// Greator (Disk index) headers
#include "greator/pq_flash_index.h"
#include "greator/aux_utils.h"
#include "greator/linux_aligned_file_reader.h"
#include "greator/utils.h"

// DiskANN (Memory index) headers
#include "diskann/index_factory.h"

namespace po = boost::program_options;

void print_full_experiment_settings(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    uint32_t R, uint32_t L, uint32_t K,
    uint32_t B, uint32_t M,
    float alpha,
    uint32_t insert_threads,
    uint32_t consolidate_threads,
    uint32_t build_threads,
    uint32_t search_threads,
    const std::string& distance_metric,
    int single_file_index,
    uint32_t beamwidth,
    uint64_t num_nodes_to_cache)
{
    const int width = 70;
    std::string line(width, '=');

    std::cout << "\n\n" << line << '\n';
    std::cout << " Full Experiment Settings\n";
    std::cout << line << '\n';

    auto print_setting = [&](const std::string& name, const auto& value)
    {
        std::ostringstream oss;
        oss << name << ": ";

        std::string val_str;
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) {
            val_str = value;
        } else if constexpr (std::is_floating_point_v<std::decay_t<decltype(value)>>) {
            std::ostringstream tmp;
            tmp.precision(2);
            tmp << std::fixed << value;
            val_str = tmp.str();
        } else {
            val_str = std::to_string(value);
        }

        oss << val_str;
        std::string line_str = oss.str();

        if ((int)line_str.size() < width)
            line_str += std::string(width - line_str.size(), ' ');

        std::cout << line_str << '\n';
    };

    print_setting("data_type", data_type);
    print_setting("data_path", data_path);
    print_setting("query_path", query_path);
    print_setting("groundtruth_path", groundtruth_path);

    print_setting("R", R);
    print_setting("L", L);
    print_setting("K", K);
    print_setting("B", B);
    print_setting("M", M);

    print_setting("alpha", alpha);

    print_setting("build_threads", build_threads);
    print_setting("insert_threads", insert_threads);
    print_setting("consolidate_threads", consolidate_threads);
    print_setting("search_threads", search_threads);

    print_setting("distance_metric", distance_metric);
    print_setting("single_file_index", single_file_index);
    print_setting("beamwidth", beamwidth);
    print_setting("num_nodes_to_cache", num_nodes_to_cache);

    std::cout << line << "\n\n";
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path;
    uint32_t R, L, K, B, M;
    uint32_t build_threads, insert_threads, consolidate_threads, search_threads;
    float alpha;
    std::string distance_metric;
    int single_file_index;
    uint32_t beamwidth;
    uint64_t num_nodes_to_cache;

    po::options_description desc;


    // Take command line parameters
    try
    {
        po::options_description desc("Allowed options");

        desc.add_options()
            ("help,h", "Print information on arguments")
        
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
        
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("L", po::value<uint32_t>(&L)->required(), "Value of L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
        
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
        
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("insert_threads", po::value<uint32_t>(&insert_threads)->required(), "Threads for inserting")
            ("consolidate_threads", po::value<uint32_t>(&consolidate_threads)->required(), "Threads for consolidation")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
        
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
        
            ("distance_metric", po::value<std::string>(&distance_metric)->default_value("l2"), "Distance metric")
            ("single_file_index", po::value<int>(&single_file_index)->default_value(0), "Single file index (0/1)")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("num_nodes_to_cache", po::value<uint64_t>(&num_nodes_to_cache)->default_value(500), "Number of nodes to cache around medoid");
    

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // Print experiment parameters
    print_full_experiment_settings(
        data_type, data_path, query_path, groundtruth_path,
        R, L, K, B, M,
        alpha,
        insert_threads, consolidate_threads, build_threads, search_threads,
        distance_metric, single_file_index, beamwidth, num_nodes_to_cache
    );

}