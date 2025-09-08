#ifndef STDVECTORUTILS_H
#define STDVECTORUTILS_H

#include <stdint.h>
#include <iostream>
#include <vector>
#include <algorithm>




namespace stdVectorUtils
{


// Check if vector contains an element
template <typename T>
bool contains(
    const std::vector<T>& vecObj,
    const T& element)
{
    // Get the iterator of first occurrence
    // of given element in vector
    auto it = std::find(
                  vecObj.begin(),
                  vecObj.end(),
                  element) ;
    return it != vecObj.end();
}

template <typename T>
void list_uniques(const std::vector<T>& source, std::vector<T>& result)
{

    typename std::vector<T>::const_iterator iter ;
    for(iter= source.begin(); iter != source.end(); ++iter)
    {
        if (not(contains(result, *iter)))
        {
            result.push_back(*iter);
        }
    }

}// list_uniques

template <typename T>
void print_vec(const std::vector<T>& source)
{
    for (uint32_t i=0;i<source.size();i++)
        {
            int key   = source[i];
            
            std::cout << key;
        }
        std::cout <<std::endl;
} // print vec


} // end namespace









#endif
