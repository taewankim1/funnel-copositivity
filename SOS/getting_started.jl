using DynamicPolynomials
@polyvar x y
p = 2*x^4 + 2*x^3*y - x^2*y^2 + 5*y^4

using SumOfSquares
using MosekTools
solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
model = SOSModel(solver)
con_ref = @constraint(model, p >= 0)
optimize!(model)
primal_status(model)

q = gram_matrix(con_ref)
sosdec = SOSDecomposition(q)