
#include "OutputSelector.h"

OutputSelector::OutputSelector(ConfigMap configMap)
{
    std::string section_start = "output_selector_";

    int n_selector = 1;
    std::string section_name = section_start + std::to_string(n_selector);
    std::string selector_type = configMap.getString(section_name, "type", "undefined");

    while (selector_type != "undefined")
    {
        std::cout << "found selector no " << n_selector << "of type " << selector_type << std::endl;

        if (selector_type == "segment")
        {

            _selectors.push_back(std::make_shared<SegmentSelector>(configMap, section_name));
        }
        else if (selector_type == "list")
        {
            _selectors.push_back(std::make_shared<ListSelector>(configMap, section_name));
        }
        else
        {
            std::cout << "found invalid selector type, stopping parsing selectors" << std::endl;
            break;
        }

        n_selector++;
        section_name = section_start + std::to_string(n_selector);
        selector_type = configMap.getString(section_name, "type", "undefined");
    }

    if (n_selector == 1)
    {
        std::cout << "found no selector, using old behavior: check for nOutput in run section" << std::endl;
        int nOutput = configMap.getInteger("run", "nOutput", 1);
        std::cout << "using nOutput = "<< nOutput << std::endl;
        _selectors.push_back(std::make_shared<SegmentSelector>(0, nOutput, -1));
    }
};

bool
OutputSelector::is_output_step(int nstep)
{
    for (auto it = _selectors.begin(); it != _selectors.end(); ++it)
    {
        if (it->get()->is_valid(nstep))
        {
            // std::cout<< it->get()->is_valid(nstep)<< std::endl;
            return true;
        }
    }

    return false;
};
