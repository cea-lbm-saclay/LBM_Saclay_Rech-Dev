#ifndef OUTPUT_SELECTOR_H_
#define OUTPUT_SELECTOR_H_

#include "ConfigMap.h"
#include "Selectors.h"
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <memory>


class OutputSelector
{
  public:
    OutputSelector(){};
    OutputSelector(ConfigMap configMap);

    bool is_output_step(int nstep);

  private:
    std::list<std::shared_ptr<SelectorBase>> _selectors;
};



#endif // OUTPUT_SELECTOR_H_
