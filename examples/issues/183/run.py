import torch

import dxtb
from dxtb.typing import DD

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

numbers = torch.tensor([8, 1, 1], device=dd["device"])
positions = torch.tensor(
    [
        [-2.95915993, 1.40005084, 0.24966306],
        [-2.1362031, 1.4795743, -1.38758999],
        [-2.40235213, 2.84218589, 1.24419946],
    ],
    requires_grad=True,
    **dd,
)

# FIX
opts = {"differentiation": "unrolled"}

par = dxtb.ParamModule(dxtb.GFN1_XTB, **dd)
par.set_differentiable("charge", "effective", "gexp")
par.set_differentiable("element", "O", "levels")
gexp = par.get("charge", "effective", "gexp")
levels_o = par.get("element", "O", "levels")

calc = dxtb.Calculator(numbers, par, opts=opts, **dd)


def get_energy_force(calc):
    energy = calc.energy(positions)
    force = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    return energy, force


# energy and AD force
energy, force = get_energy_force(calc)

# AD gradient w.r.t. params
de_dparam = torch.autograd.grad(energy, (gexp, levels_o), retain_graph=True)
dfnorm_dparam = torch.autograd.grad(torch.norm(force), (gexp, levels_o))

# Numerical gradient w.r.t. params
dparam = 2e-6
with torch.no_grad():
    orig_gexp = gexp.item()
    gexp.fill_(orig_gexp + dparam / 2)
energy1, force1 = get_energy_force(calc)
with torch.no_grad():
    gexp.fill_(orig_gexp - dparam / 2)
energy2, force2 = get_energy_force(calc)
with torch.no_grad():
    gexp.fill_(orig_gexp)

print(
    f"dE / dgexp = {de_dparam[0].item()} (AD) {(energy1-energy2)/dparam} (Numerical)"
)
print(
    f"d|F| / dgexp = {dfnorm_dparam[0].item()} (AD) {(torch.norm(force1)-torch.norm(force2)).item()/dparam} (Numerical)"
)

with torch.no_grad():
    orig_level = levels_o[0].item()
    levels_o[0] = orig_level + dparam / 2
energy1, force1 = get_energy_force(calc)
with torch.no_grad():
    levels_o[0] = orig_level - dparam / 2
energy2, force2 = get_energy_force(calc)
with torch.no_grad():
    levels_o[0] = orig_level

print(
    f"dE / dlevels_O[0] = {de_dparam[1][0].item()} (AD) {(energy1-energy2)/dparam} (Numerical)"
)
print(
    f"d|F| / dlevels_O[0] = {dfnorm_dparam[1][0].item()} (AD) {(torch.norm(force1)-torch.norm(force2)).item()/dparam} (Numerical)"
)
