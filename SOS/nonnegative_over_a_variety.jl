using DynamicPolynomials
using SumOfSquares
@polyvar x y
S = @set x^2 + y^2 == 1
using MosekTools
model = SOSModel(Mosek.Optimizer)
set_silent(model)
con_ref = @constraint(model, 1 - y^2 >= 0, domain = S)
optimize!(model)

sos_decomposition(con_ref, 1e-6)