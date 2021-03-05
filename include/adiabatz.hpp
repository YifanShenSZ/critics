#ifndef adiabatz_hpp
#define adiabatz_hpp

#include <torch/torch.h>

namespace adiabatz {

extern size_t NStates;

void initialize_adiabatz(const std::vector<std::string> & args);

at::Tensor compute_energy(const at::Tensor & r);

std::tuple<at::Tensor, at::Tensor> compute_energy_dHa(const at::Tensor & r);

at::Tensor compute_ddHa(const at::Tensor & r);

} // namespace adiabatz

#endif