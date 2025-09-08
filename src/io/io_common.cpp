#include <ctime> // for std::time_t, std::tm, std::localtime
#include <iomanip> // for std::put_time (only g++ >= 5)
#include <sstream>

#include "io_common.h"

namespace lbm_saclay {

// =======================================================
// =======================================================
std::string format_index(const int index, const int fmt_width)
{
    std::ostringstream fmt;
    fmt.width(fmt_width);
    fmt.fill('0');
    fmt << index;
    return fmt.str();
}

// =======================================================
// =======================================================
std::string current_date()
{

    /* get current time */
    std::time_t now = std::time(nullptr);

    /* Format and print the time, "ddd yyyy-mm-dd hh:mm:ss zzz" */
    std::tm tm = *std::localtime(&now);

    // old versions of g++ don't have std::put_time,
    // so we provide a slight work arround
#if defined(__GNUC__) && (__GNUC__ < 5)

    char foo[64];

    std::strftime(foo, sizeof(foo), "%Y-%m-%d %H:%M:%S %Z", &tm);
    return std::string(foo);

#else

    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %Z");

    return ss.str();

#endif

} // current_date

// =======================================================
// =======================================================
int2str_t build_var_to_write_map(const LBMParams& params, const ConfigMap& configMap, int2str_t avail_ids)
{
    int2str_t map;
    // Read parameter file and get the list of variable field names
    // we wish to write
    // variable names must be comma-separated (no quotes !)
    // e.g. write_variables=rho,vx,vy,unknown
    std::string write_variables = configMap.getString("output", "write_variables", "rho");

    //invert the id->name map
    str2int_t avail_names;
    for (const std::pair<const int, std::string>& n : avail_ids) {
        //~ print_key_value(n.first, n.second);
        avail_names[n.second] = n.first;
    }

    // now tokenize
    std::istringstream iss(write_variables);
    std::string token;
    while (std::getline(iss, token, ',')) {

        // check if token is valid, i.e. present in avail_names
        auto got = avail_names.find(token);
        //~ std::cout << " " << token << " " << got->second << "\n";

        // if token is valid, we insert it into map
        if (got != avail_names.end()) {
            map[got->second] = token;
        }
    }

    return map;

} // build_var_to_write_map

} // namespace lbm_saclay
