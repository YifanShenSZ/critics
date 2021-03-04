#include <Foptim/BFGS.hpp>

#include "adiabatz.hpp"
#include "global.hpp"

namespace {

at::Tensor init_guess_;

void energy(double & energy, const double * intgeom, const int32_t & intdim) {
    at::Tensor q = at::from_blob(const_cast<double *>(intgeom), intdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e = adiabatz::compute_energy(r);
    energy = e[target_state].item<double>();
}

void energy_grad(double & energy, double * grad, const double * intgeom, const int32_t & intdim) {
    at::Tensor q = at::from_blob(const_cast<double *>(intgeom), intdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e, dHa;
    std::tie(e, dHa) = adiabatz::compute_energy_dHa(r);
    at::Tensor intgrad = intcoordset->gradient_cart2int(r, dHa[target_state][target_state]);
    energy = e[target_state].item<double>();
    std::memcpy(grad, intgrad.data_ptr<double>(), intdim * sizeof(double));
}

void Hessian(double * Hessian, const double * intgeom, const int32_t & intdim) {
    at::Tensor q = at::from_blob(const_cast<double *>(intgeom), intdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e, dHa;
    std::tie(e, dHa) = adiabatz::compute_energy_dHa(r);
    at::Tensor cartHess = adiabatz::compute_ddHa(r)[target_state][target_state];
    at::Tensor intHess = intcoordset->Hessian_cart2int(r, dHa[target_state][target_state], cartHess);
    std::memcpy(Hessian, intHess.data_ptr<double>(), intdim * intdim * sizeof(double));
}

}

at::Tensor search_minimum(const at::Tensor & _init_guess) {
    init_guess_ = _init_guess;
    at::Tensor q = (*intcoordset)(_init_guess);
    Foptim::BFGS(energy, energy_grad, Hessian, q.data_ptr<double>(), q.size(0),
                 20, 100, 1e-6, 1e-6);
    at::Tensor r = int2cart(q, _init_guess, intcoordset);
    return r;
}