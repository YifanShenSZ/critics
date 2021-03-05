#ifndef int2cart_hpp
#define int2cart_hpp

#include <tchem/intcoord.hpp>

at::Tensor int2cart(const at::Tensor & q, const at::Tensor & init_guess,
const std::shared_ptr<tchem::IC::IntCoordSet> & _intcoordset);

#endif