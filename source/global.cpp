#include "global.hpp"

std::shared_ptr<tchem::IC::IntCoordSet> intcoordset;

size_t target_state;

std::shared_ptr<Fixed_intcoord> fixed_intcoord;





Fixed_intcoord::Fixed_intcoord() {}
Fixed_intcoord::Fixed_intcoord(const std::vector<size_t> & _fixed_coords, const at::Tensor & init_q)
: fixed_coords_(_fixed_coords) {
    for (size_t i = 0; i < intcoordset->size(); i++)
    if (std::find(_fixed_coords.begin(), _fixed_coords.end(), i) == _fixed_coords.end())
    free_coords_.push_back(i);

    for (const size_t & fixed_coord : _fixed_coords)
    fixed_values_.push_back(init_q[fixed_coord].item<double>());
}
Fixed_intcoord::~Fixed_intcoord() {}

at::Tensor Fixed_intcoord::vector_free2total(const at::Tensor & V_free) const {
    at::Tensor V = V_free.new_empty(intcoordset->size());
    for (size_t i = 0; i < fixed_coords_.size(); i++)
    V[fixed_coords_[i]].fill_(fixed_values_[i]);
    for (size_t i = 0; i < free_coords_.size(); i++)
    V[free_coords_[i]].copy_(V_free[i]);
    return V;
}
at::Tensor Fixed_intcoord::vector_total2free(const at::Tensor & V) const {
    at::Tensor V_free = V.new_empty(free_coords_.size());
    for (size_t i = 0; i < free_coords_.size(); i++)
    V_free[i].copy_(V[free_coords_[i]]);
    return V_free;
}

at::Tensor Fixed_intcoord::matrix_total2free(const at::Tensor & M) const {
    int64_t NFree = free_coords_.size();
    at::Tensor M_free = M.new_empty({NFree, NFree});
    for (size_t i = 0; i < NFree; i++)
    for (size_t j = 0; j < NFree; j++)
    M_free[i][j].copy_(M[free_coords_[i]][free_coords_[j]]);
    return M_free;
}