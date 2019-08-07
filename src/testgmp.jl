using Convex
using SDPA_GMP
using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities
using SemidefiniteOptInterface
const SDOI = SemidefiniteOptInterface

using SparseArrays
# using SDPA

using LinearAlgebra

eye(n) = Matrix(big(1.0)*I, n, n)

# x = [1, 2, 3]
# P = Variable(3, 3)
# p1 = Problem{BigFloat}(:minimize, matrixfrac(x, P), [P <= big(2)*eye(3), P >= big(0.5) * eye(3)])

x = Semidefinite(3)
p = Problem{BigFloat}(:minimize, sumlargesteigs(x, 2), [x >= big(1)])
#
# mock1 = SDPA_GMP.sdpa_gmp_binary_solve(p1)
#
Y = Variable(5,5)
X = -1*rand(BigFloat, 5, 5)
p2 = Problem{BigFloat}(:minimize, tr(Y), [ diag(Y)[2:5] == diag(X)[2:5], Y[1,1] == big(0.0) ])
#

x = Variable(1)
y = Variable(1)
p2 = Problem{BigFloat}(:maximize, x + y, [ x <= big"1.0" , y <= big"2.0"])

mock = Convex.solve!(p, SDPA_GMP.SDPAGMPoptimizer(BigFloat, verbose = true));

# mock2 = SDPA_GMP.sdpa_gmp_binary_solve(p2)

# E12, E21 = ComplexVariable(2, 2), ComplexVariable(2, 2)
# s1, s2 = [big"0.25" big"-0.25"*im; big"0.25"*im big"0.25"], [big"0.5" big"0.0"; big"0.0" big"0.0"]
# p3 = Problem{BigFloat}(:minimize, real(tr(E12 * (s1 + big"2.0000009" * s2) + E21 * (s2 + 2 * s1))), [E12 ⪰ 0, E21 ⪰ 0, E12 + E21 == Diagonal([big"1.0", big"1.0"]) ])
#
# mock3 = SDPA_GMP.sdpa_gmp_binary_solve(p3)
x = Variable(Positive())
y = Semidefinite(3)
p = Problem{BigFloat}(:minimize, nuclearnorm(y), y[2,1]<=4, y[2,2]>=3, y[3,3]<=2)
solve!(p, SDPA_GMP.SDPAGMPoptimizer(BigFloat, verbose = true, normal_sdpa = true));