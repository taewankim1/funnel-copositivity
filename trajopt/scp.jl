# include("dynamics.jl")
# include("constraint.jl")
# include("costfunction.jl")
# include("scaling.jl")
# include("utils.jl")
# include("discretize.jl")

using LinearAlgebra
using Printf
using JuMP
using MosekTools
using Clarabel
using Gurobi

mutable struct Solution
    x::Matrix{Float64}
    u::Matrix{Float64}
    dt::Vector{Float64}
    # t::Vector{Float64}

    A::Array{Float64,3}
    Bm::Array{Float64,3}
    Bp::Array{Float64,3}
    smat::Matrix{Float64}
    z::Matrix{Float64}

    xi::Vector{Float64}
    xf::Vector{Float64}

    t::Any
    tprop::Any
    xprop::Any
    function Solution(N::Int64,ix::Int64,iu::Int64)
        x = zeros(ix,N+1)
        u = zeros(iu,N+1)
        dt = zeros(N)
        # t = [0];
        # for i = 1:N
        #     push!(t,sum(dt[1:i]))
        # end
        A = zeros(ix,ix,N)
        Bm = zeros(ix,iu,N)
        Bp = zeros(ix,iu,N)
        smat = zeros(size(x))
        z = zeros(size(x))

        xi = zeros(ix)
        xf = zeros(ix)
        new(x,u,dt,A,Bm,Bp,smat,z,xi,xf)
    end
    # function get_t()
    # end
end

function update_t(sol::Solution)
    sol.t = [0.0]
    N = length(sol.dt)
    for i = 1:N
        push!(sol.t,sum(sol.dt[1:i]))
    end
end

abstract type SCP end

struct PTR <: SCP
    dynamics::Dynamics  # system dynamics
    constraint::Vector{Constraint}  # constraints
    scaling::Any  # struct having parameters for scaling
    solution::Solution

    N::Int64  # number of subintervals (number of node - 1)
    # min_time_scale::Float64  # minimum time for subinterval
    # max_time_scale::Float64  # maximum time for subinterval
    tf::Float64  # final time (time horizon)
    w_tf::Float64  # weight for final time
    w_c::Float64  # weight for input penalty (or all others except time)
    w_rate::Float64  # weight for control rate
    w_param::Float64  # weight for energy bound
    w_vc::Float64  # weight for virtual control
    w_tr::Float64  # weight for trust-region
    tol_tr::Float64  # tolerance for trust-region
    tol_vc::Float64  # tolerance for virtual control
    tol_dyn::Float64  # tolerance for dynamics error
    tr_norm::Any  # choice for trust-region regularization (quadratic, 2-norm, 1-norm..)
    max_iter::Int64  # maximum iteration
    type_tf_free::String
    verbosity::Bool

    function PTR(N::Int, tf::Float64, max_iter::Int,
        dynamics::Dynamics, constraint::Vector{T}, scaling::Scaling,
        w_tf::Float64, w_c::Float64, w_rate::Float64, w_param::Float64, w_vc::Float64, w_tr::Float64,
        tol_vc::Float64, tol_tr::Float64, tol_dyn::Float64, tr_norm::Any,
        verbosity::Bool;type_tf_free::String="tf_fixed") where T <: Constraint

        ix = dynamics.ix
        iu = dynamics.iu
        solution = Solution(N,ix,iu)

        return new(dynamics, constraint, scaling, solution, N, tf,
            w_tf, w_c, w_rate, w_param, w_vc, w_tr, tol_tr,
            tol_vc, tol_dyn, tr_norm, max_iter, type_tf_free, verbosity)
    end
end

function uniform_fixed!(ptr::SCP,model::Model)
    N = ptr.N
    S_sigma = ptr.scaling.S_sigma 
    dt = [ptr.tf/N/S_sigma for i in 1:N]
    i = 1
    min_dt = ptr.scaling.min_dt
    max_dt = ptr.scaling.max_dt
    @constraint(model,S_sigma*dt[i] >= 0)
    @constraint(model,S_sigma*dt[i] >= min_dt)
    @constraint(model,S_sigma*dt[i] <= max_dt)
    return dt
end

function uniform_mesh!(ptr::SCP,model::Model)
    N = ptr.N
    @variable(model,Δt)
    dt = [Δt for i in 1:N]

    S_sigma = ptr.scaling.S_sigma 
    min_dt = ptr.scaling.min_dt
    max_dt = ptr.scaling.max_dt

    i = 1
    @constraint(model,S_sigma*dt[i] >= 0)
    @constraint(model,S_sigma*dt[i] >= min_dt)
    @constraint(model,S_sigma*dt[i] <= max_dt)
    return dt
end

function uniform_two_phase!(ptr::SCP,model::Model)
    N = ptr.N
    N1 = ptr.constraint[end].N1 # assume the multiphase constraint given as the last element
    N2 = ptr.constraint[end].N2
    @assert(N == N1 + N2)
    @variable(model,Δt1)
    @variable(model,Δt2)
    dt1 = [Δt1 for i in 1:N1]
    dt2 = [Δt2 for i in 1:N2]
    dt = vcat(dt1,dt2)

    S_sigma = ptr.scaling.S_sigma 
    min_dt = ptr.scaling.min_dt
    max_dt = ptr.scaling.max_dt

    for Δt in [Δt1,Δt2]
        @constraint(model,S_sigma*Δt >= 0)
        @constraint(model,S_sigma*Δt >= min_dt)
        @constraint(model,S_sigma*Δt <= max_dt)
    end
    return dt
end

function uniform_five_phase!(ptr::SCP,model::Model)
    N = ptr.N
    N1 = ptr.constraint[end].N1 # assume the multiphase constraint given as the last element
    N2 = ptr.constraint[end].N2
    N3 = ptr.constraint[end].N3
    N4 = ptr.constraint[end].N4
    N5 = ptr.constraint[end].N5
    N6 = ptr.constraint[end].N6
    @assert(N + 1 == N6)
    @variable(model,Δt1)
    @variable(model,Δt2)
    @variable(model,Δt3)
    @variable(model,Δt4)
    @variable(model,Δt5)
    dt1 = [Δt1 for i in 1:N2-N1]
    dt2 = [Δt2 for i in 1:N3-N2]
    dt3 = [Δt3 for i in 1:N4-N3]
    dt4 = [Δt4 for i in 1:N5-N4]
    dt5 = [Δt5 for i in 1:N6-N5]
    dt = vcat(dt1,dt2,dt3,dt4,dt5)

    S_sigma = ptr.scaling.S_sigma 
    min_dt = ptr.scaling.min_dt
    max_dt = ptr.scaling.max_dt

    for Δt in [Δt1,Δt2,Δt3,Δt4,Δt5]
        @constraint(model,S_sigma*Δt >= 0)
        @constraint(model,S_sigma*Δt >= min_dt)
        @constraint(model,S_sigma*Δt <= max_dt)
    end
    return dt
end

function nonuniform_mesh!(ptr::SCP,model::Model)
    N = ptr.N
    @variable(model,dt[1:N])
    S_sigma = ptr.scaling.S_sigma 
    min_dt = ptr.scaling.min_dt
    max_dt = ptr.scaling.max_dt

    for i in 1:N
        @constraint(model,S_sigma*dt[i] >= 0)
        @constraint(model,S_sigma*dt[i] >= min_dt)
        @constraint(model,S_sigma*dt[i] <= max_dt)
    end
    return dt
end

function cvxopt(ptr::PTR,solver::Any,solver_env::Any)
    ix = ptr.dynamics.ix
    iu = ptr.dynamics.iu
    N = ptr.N

    xbar = ptr.solution.x
    ubar = ptr.solution.u
    dtbar = ptr.solution.dt

    Sx = ptr.scaling.Sx
    iSx = ptr.scaling.iSx
    sx = ptr.scaling.sx
    Su = ptr.scaling.Su
    iSu = ptr.scaling.iSu
    su = ptr.scaling.su
    S_sigma = ptr.scaling.S_sigma

    # cvx model
    if solver == "Mosek"
        model = Model(Mosek.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0) # Turn off verbosity for Mosek
    elseif solver == "Clarabel"
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", false) # Turn off verbosity for Mosek
    elseif solver == "Gurobi"
        model = Model(() -> Gurobi.Optimizer(solver_env))
        set_optimizer_attribute(model, "OutputFlag", 0)
    else
        println("You should select Mosek, Clarabel, Gurobi")
    end
    # println(typeof(model))

    # cvx variables (scaled)
    @variable(model, xcvx[1:ix,1:N+1])
    @variable(model, ucvx[1:iu,1:N+1])
    @variable(model, vc[1:ix,1:N])
    @variable(model, vc_t[1:N])

    # constraints on dt
    if ptr.type_tf_free == "tf_free"
        dt = uniform_mesh!(ptr,model)
    elseif ptr.type_tf_free == "tf_fixed"
        dt = uniform_fixed!(ptr,model)
    elseif ptr.type_tf_free == "two_phase"
        dt = uniform_two_phase!(ptr,model)
    elseif ptr.type_tf_free == "five_phase"
        dt = uniform_five_phase!(ptr,model)
    end
    # println(model)

    # scale reference trajectory
    xbar_scaled = zeros(ix,N+1)
    ubar_scaled = zeros(iu,N+1)
    dtbar_scaled = zeros(N)

    for i in 1:N
        xbar_scaled[:,i] .= iSx*(xbar[:,i]-sx)
        ubar_scaled[:,i] .= iSu*(ubar[:,i]-su)
        dtbar_scaled[i] = 1/S_sigma*dtbar[i]
    end
    xbar_scaled[:,N+1] .= iSx*(xbar[:,N+1]-sx)
    ubar_scaled[:,N+1] .= iSu*(ubar[:,N+1]-su)

    # initial and boundary
    initial_condition!(ptr.dynamics,model,Sx*xcvx[:,1]+sx,ptr.solution.xi)
    final_condition!(ptr.dynamics,model,Sx*xcvx[:,N+1]+sx,ptr.solution.xf,uN=Sx*xcvx[:,N]+sx)
    
    # constraints
    N_constraint = size(ptr.constraint,1)
    for i in 1:N+1
        for j in 1:N_constraint
            impose!(ptr.constraint[j],model,Sx*xcvx[:,i]+sx,Su*ucvx[:,i]+su,xbar[:,i],ubar[:,i],idx=i)
        end
    end

    # # Hard coding (intermediate waypoint)
    # @constraint(model, (Sx*xcvx[:,11]+sx)[1:2] == [4;8])

    # Dynamics
    for i in 1:N
        @constraint(model,xcvx[:,i+1] == iSx*ptr.solution.A[:,:,i]*(Sx*xcvx[:,i] + sx)
            +iSx*ptr.solution.Bm[:,:,i]*(Su*ucvx[:,i]+su)
            +iSx*ptr.solution.Bp[:,:,i]*(Su*ucvx[:,i+1]+su)
            +iSx*ptr.solution.smat[:,i]*S_sigma*dt[i]
            +iSx*ptr.solution.z[:,i]
            +vc[:,i]
            )
    end

    # cost
    cost_tf = [S_sigma*dt[i] for i in 1:N]
    l_tf = sum(cost_tf)

    for i in 1:N
        @constraint(model, [vc_t[i]; vc[:,i]] in MOI.NormOneCone(1 + ix))
    end
    cost_vc = [vc_t[i] for i in 1:N]
    l_vc = sum(cost_vc)
    
    cost_tr = [dot(xcvx[:,i]-xbar_scaled[:,i],xcvx[:,i]-xbar_scaled[:,i]) + dot(ucvx[:,i]-ubar_scaled[:,i],ucvx[:,i]-ubar_scaled[:,i]) for i in 1:N+1]
    l_tr = sum(cost_tr)

    cost_running = [get_cost(ptr.dynamics,Sx*xcvx[:,i]+sx,Su*ucvx[:,i]+su,i,N) for i in 1:N+1]
    l_c = sum(cost_running)

    cost_rate = [dot(ucvx[:,i+1] - ucvx[:,i],ucvx[:,i+1] - ucvx[:,i]) for i in 1:N]
    l_rate = sum(cost_rate)

    l_all = ptr.w_tf * l_tf + ptr.w_c * l_c + ptr.w_rate * l_rate + ptr.w_tr * l_tr + ptr.w_vc * l_vc

    l_normalize = l_all / 1e3

    @objective(model, Min, l_normalize)
    optimize!(model)
    # @assert is_solved_and_feasible(model)

    for i in 1:N+1
        ptr.solution.x[:,i] .= Sx*value.(xcvx[:,i]) + sx
        ptr.solution.u[:,i] .= Su*value.(ucvx[:,i]) + su
    end
    ptr.solution.dt .= S_sigma*value.(dt)
    return value(l_tf),value(l_rate),value(l_vc),value(l_tr),value(l_c),value(l_all)
end

function run(ptr::SCP,x0::Matrix,u0::Matrix,dt0::Vector,xi::Vector,xf::Vector,solver::String)
    ptr.solution.x .= x0
    ptr.solution.u .= u0
    ptr.solution.dt .= dt0

    ptr.solution.xi .= xi
    ptr.solution.xf .= xf

    ix = ptr.dynamics.ix
    iu = ptr.dynamics.iu
    N = ptr.N

    if solver == "Gurobi"
        solver_env = Gurobi.Env()
    else
        solver_env = nothing
    end

    for iteration in 1:ptr.max_iter
        # discretization & linearization
        ptr.solution.A,ptr.solution.Bm,ptr.solution.Bp,ptr.solution.smat,ptr.solution.z,_ = discretize_foh(ptr.dynamics,
            ptr.solution.x[:,1:N],ptr.solution.u,ptr.solution.dt)
        
        # solve subproblem
        c_tf,c_rate,c_vc,c_tr,c_input,c_all = cvxopt(ptr,solver,solver_env);

        # multiple shooting
        xfwd,ptr.solution.tprop,ptr.solution.xprop = propagate_multiple_FOH(ptr.dynamics,ptr.solution.x,ptr.solution.u,ptr.solution.dt)
        dyn_error = maximum(norm.(eachcol(xfwd - ptr.solution.x), 2))

        # accept the step
        # We aready did

        # print
        if ptr.verbosity == true && iteration == 1
            println("+--------------------------------------------------------------------------------------------------+")
            println("|                                   ..:: Penalized Trust Region ::..                               |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
            println("| iter. |    cost    |    tof    |   main    |   rate    |  param  | log(vc) | log(tr)  | log(dyn) |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
        end
        # println(c_vc,"/",c_tr,"/",dyn_error)
        @printf("|%-2d     |%-7.2f     |%-7.3f   |%-7.3f    |%-7.3f    |%-5.3f    |%-5.1f    | %-5.1f    |%-5.1e   |\n",
            iteration,
            c_all,c_tf,c_input,c_rate,
            -1,
            log10(abs(c_vc)), log10(abs(c_tr)), log10(abs(dyn_error)))

        flag_vc::Bool = c_vc < ptr.tol_vc
        flag_tr::Bool = c_tr < ptr.tol_tr
        flag_dyn::Bool = dyn_error < ptr.tol_dyn

        if flag_vc && flag_tr && flag_dyn
            println("+--------------------------------------------------------------------------------------------------+")
            println("Converged!")
            break
        end
    end
    update_t(ptr.solution)
end



