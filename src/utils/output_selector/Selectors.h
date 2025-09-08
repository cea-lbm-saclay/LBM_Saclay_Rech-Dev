#ifndef SELECTORS_H_
#define SELECTORS_H_

#include "../config/ConfigMap.h"
#include <iostream>
#include <string>
#include <vector>

class SelectorBase
{
  public:
    SelectorBase(){};
    SelectorBase(ConfigMap config_map, std::string section_name){};
    virtual bool is_valid(int i) { return false; };
};

class SegmentSelector : public SelectorBase
{
  public:
    SegmentSelector(ConfigMap config_map, std::string section_name);
    
    SegmentSelector(int n_start, int n_step, int n_end);

    bool is_valid(int i) override;

  private:
    int n_start;
    int n_step;
    int n_end;
};

class ListSelector : public SelectorBase
{
  public:
    ListSelector(ConfigMap config_map, std::string section_name);

    bool is_valid(int i) override;

  private:
    std::vector<int> list;
};

#endif
