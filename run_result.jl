include("./trajopt/utils.jl")
include("./trajopt/dynamics.jl")
include("./funlopt/funl_dynamics.jl")
include("./funlopt/funl_utils.jl")
include("./funlopt/funl_constraint.jl")
include("./trajopt/scaling.jl")

# Set default plot settings for academic paper
default(fontfamily="Times",  # Use Times New Roman font
        titlefont=14,        # Title font size
        guidefont=12,        # Label (guide) font size
        tickfont=10,         # Tick font size
        legendfont=12,       # Legend font size
        grid=true)          # Remove grid lines for a cleaner look


# load nominal trajectory
using JLD2, FileIO
filename = "./data/traj_result" 
@load filename my_dict
xnom = my_dict["x"]
unom = my_dict["u"]
tnom = my_dict["t"];
N = size(xnom,2) - 1
dtnom = zeros(N)
for i in 1:N
    dtnom[i] = tnom[i+1]-tnom[i]
end

function get_H_obs(rx,ry)
    return diagm([1/rx,1/ry])
end
c_list = []
H_list = []
push!(c_list, [4.0, 2.5])
push!(H_list, get_H_obs(1.5, 4.0))

vmax = 2.0
vmin = 0.0
wmax = 1.5
wmin = -1.5
list_const = Vector{FunnelConstraint}()
push!(list_const,InputConstraint([1;0],vmax))
push!(list_const,InputConstraint([-1;0],-vmin))
push!(list_const,InputConstraint([0; 1],wmax))
push!(list_const,InputConstraint([0; -1],-wmin))
for i = 1:length(c_list)
    push!(list_const,ObstacleAvoidance(H_list[i],c_list[i]))
end

ix = 3
iu = 2
iw = 2
dynamics = Unicycle()

A = zeros(ix,ix,N+1)
B = zeros(ix,iu,N+1)
F = zeros(ix,iw,N+1)
for idx in 1:N+1
    A[:,:,idx],B[:,:,idx],F[:,:,idx] = diff_ABF(dynamics,xnom[:,idx],unom[:,idx])
end

Qmax = zeros(ix,ix,N+1)
for idx in 1:N+1
    Qmax[:,:,idx] .= diagm([2^2,2^2,deg2rad(20)^2])
end
Rmax = get_Rmax_unicycle(unom,N,vmax,vmin,wmax,wmin)
;

include("./funlopt/funl_utils.jl")

gamma_est = Lipschitz_estimation_around_traj(N,100,xnom,unom,dynamics,Qmax,Rmax)
gamma = gamma_est[1:N]

using Interpolations

u_fit = [LinearInterpolation(tnom, unom[idx,:],extrapolation_bc=Flat()) for idx in 1:iu]
A_fit = [[LinearInterpolation(tnom, A[i,j,:], extrapolation_bc=Flat()) for j in 1:ix] for i in 1:ix ]
B_fit = [[LinearInterpolation(tnom, B[i,j,:], extrapolation_bc=Flat()) for j in 1:iu] for i in 1:ix ]
F_fit = [[LinearInterpolation(tnom, F[i,j,:], extrapolation_bc=Flat()) for j in 1:iw] for i in 1:ix ];


function model_wrapper!(f,x,p,t)
    um = p[1]
    up = p[2]
    dt = p[3]
    alpha = (dt - t) / dt
    beta = t / dt
    u1 = alpha*um + beta*up
    f .= forward(dynamics,x,u1)
end

N_sampling = 20
beta = []
for i = 1:N
    tspan = (0,tnom[i+1]-tnom[i])
    saveat = range(0, stop=tnom[i+1]-tnom[i], length=N_sampling)
    prob = ODEProblem(model_wrapper!,xnom[:,i],tspan,(unom[:,i],unom[:,i+1],tnom[i+1]-tnom[i]),saveat = saveat)
    sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9;verbose=false);
    @assert(isapprox(sol.u[end],xnom[:,i+1];atol=0.0001))
    delta = []
    for idx_sample = 1:N_sampling
        t_eval = sol.t[idx_sample] + tnom[i]
        x_eval = sol.u[idx_sample]
        u_eval = get_u_interp(t_eval)
        A_eval,B_eval,F_eval = diff_ABF(dynamics,x_eval,u_eval)
        eA = get_ABF_interp(t_eval,A_fit,ix,ix) .- A_eval
        eB = get_ABF_interp(t_eval,B_fit,ix,iu) .- B_eval
        eF = get_ABF_interp(t_eval,F_fit,ix,iw) .- F_eval
        delta_ = [eA eB eF]
        push!(delta,opnorm(delta_,2))
    end
    push!(beta,maximum(delta))
end

# dynamics.C = [1.0I(ix);zeros(iu,ix);zeros(iw,ix);dynamics.Co]
# dynamics.D = [zeros(ix,iu);1.0I(iu);zeros(iw,iu);dynamics.Do]
# dynamics.E = [1.0I(ix) dynamics.Eo]
# dynamics.G = [zeros(ix,iw);zeros(iu,iw);1.0I(iw);dynamics.Go];
# dynamics.idelta = 3

C1 = [0 0 1;0 0 0]
D1 = [1 0;0 0]
G1 = [0 0;0 0]
E1 = [1 0;0 1;0 0]

dynamics.C = [C1;dynamics.Co]
dynamics.D = [D1;dynamics.Do]
dynamics.E = [E1 dynamics.Eo]
dynamics.G = [G1;dynamics.Go];

dynamics.ir = 2 + dynamics.iq
dynamics.ip = 2 + dynamics.iphi
dynamics.idelta = 2

xmin = [0;0;0];
xmax = [1;1;pi];
umin = [0;0];
umax = [1;1];
scaler = Scaling(xmin, xmax, umin, umax, 1e3,1e3)

include("./funlopt/funl_synthesis.jl")

# lambda_w = 2.0
list_lambda_w = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
list_solve_time = []
cost_lambda_w = []
list_fs = []
println("============ Line search for lambda_w ==============")
for lambda_w in list_lambda_w
        fs = FunnelSynthesis(N,lambda_w,dynamics,list_const,scaler,verbosity=false,flag_copositivity_type=1)
        cost,solve_time = run!(fs,
        gamma,
        beta,
        xnom,unom,tnom,Qmax,Rmax,"Mosek")
        println("lambda_w: ",lambda_w, " cost: ",abs(cost) < 1e-6 ? "diverged" : cost)
        push!(cost_lambda_w,cost)
        push!(list_fs,fs)
        push!(list_solve_time,solve_time)
end
idx_lambda_w = 1:length(list_lambda_w)
idx_converged = abs.(cost_lambda_w) .>= 1e-6
cost_lambda_w_converged = cost_lambda_w[idx_converged]
idx_fs = argmin(cost_lambda_w_converged)
idx_min = idx_lambda_w[idx_converged][idx_fs]

lambda_w = list_lambda_w[idx_min]
cost = cost_lambda_w[idx_min]
fs = list_fs[idx_min]
solve_time = list_solve_time[idx_min]

println(lambda_w," is picked")

println("======== First copositive condition ========")

println("lambda_w: ",lambda_w, " cost: ",cost," solve time ",solve_time)

fs2 = FunnelSynthesis(N,lambda_w,dynamics,list_const,scaler,verbosity=false,flag_copositivity_type=2)
cost,solve_time = run!(fs2,
gamma,
beta,
xnom,unom,tnom,Qmax,Rmax,"Mosek")

println("======== Second copositive condition ========")

println("lambda_w: ",lambda_w, " cost: ",cost," solve time ",solve_time)

Qnom_,Ynom_,lamnom_ = fs.solution.Q,fs.solution.Y,fs.solution.lam
Qnom,Ynom,lamnom = fs2.solution.Q,fs2.solution.Y,fs2.solution.lam

funl_dict = Dict("Q1" => Qnom_, "Y1" => Ynom_, "lam1" => lamnom_, "Q2" => Qnom, "Y2" => Ynom, "lam" => lamnom)

using JLD2, FileIO
@save "./data/funnel_result" funl_dict

p2 = Plots.plot(; size=(500,500))
plot!(xnom[1,:],xnom[2,:],aspect_ratio=:equal,c=:deepskyblue3,linestyle=:dash,linewidth=1.5,label=nothing)
for idx in 1:N+1
    label = nothing
    if idx == 1
        label = "funnel"
    end
    plot_ellipse(p2,Qnom_[:,:,idx],xnom[:,idx],"deepskyblue3",linewidth=1.5,label=label,fill=true)
end
for (idx,(ce, H)) in enumerate(zip(c_list, H_list))
    label = nothing
    if idx == 1
        label = "obstacle"
    end
    plot_ellipse(p2,inv(H)*inv(H),ce,"red3",label=label)
end
savefig("./funnels.png")