#ifndef global_hpp
#define global_hpp

#include <tchem/intcoord.hpp>

#include "int2cart.hpp"

extern std::shared_ptr<tchem::IC::IntCoordSet> intcoordset;

extern size_t target_state;

class Fixed_intcoord {
    private:
        std::vector<size_t> fixed_coords_;

        std::vector<size_t> free_coords_;
        std::vector<double> fixed_values_;
    public:
        Fixed_intcoord();
        Fixed_intcoord(const std::vector<size_t> & _fixed_coords, const at::Tensor & init_q);
        ~Fixed_intcoord();

        at::Tensor vector_free2total(const at::Tensor & V_free) const;
        at::Tensor vector_total2free(const at::Tensor & V) const;

        at::Tensor matrix_total2free(const at::Tensor & M) const;
};

extern std::shared_ptr<Fixed_intcoord> fixed_intcoord;

#endif