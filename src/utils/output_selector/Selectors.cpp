#include "../../stdVectorUtils.h"
#include "Selectors.h"

SegmentSelector::SegmentSelector(ConfigMap config_map, std::string section_name)
  : SelectorBase()
{
    n_start = config_map.getInteger(section_name, "n_start", 0);
    n_step = config_map.getInteger(section_name, "n_step", 1);
    n_end = config_map.getInteger(section_name, "n_end", 0);
};

SegmentSelector::SegmentSelector(int n_start, int n_step, int n_end)
  : SelectorBase()
  , n_start(n_start)
  , n_step(n_step)
  , n_end(n_end){};

bool
SegmentSelector::is_valid(int i)
{
    return (i >= n_start and (n_end < 0 or i < n_end) and ((i - n_start) % n_step) == 0);
}


ListSelector::ListSelector(ConfigMap config_map, std::string section_name)
  : SelectorBase()
{
        std::string steps = config_map.getString(section_name, "steps", "0");


        std::string delimiter = ",";
        size_t pos = 0;
        std::string token;
        while ((pos = steps.find(delimiter)) != std::string::npos)
        {
            token = steps.substr(0, pos);
            std::cout << token << std::endl;

            list.push_back(std::stoi(token));
            steps.erase(0, pos + delimiter.length());
        }
};


bool
ListSelector::is_valid(int i)
{
    return (stdVectorUtils::contains(list, i));
}
