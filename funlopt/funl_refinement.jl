using LinearAlgebra
using Printf
using JuMP
using MosekTools
using Clarabel

include("funl_dynamics.jl")
include("funl_utils.jl")
include("funl_constraint.jl")
include("funl_ctcs.jl")
include("funl_synthesis.jl")
include("../trajopt/dynamics.jl")
include("../trajopt/scaling.jl")


struct FunnelRefinement
    dynamics::Dynamics
    funl_dynamics::FunnelDynamics
    funl_constraint::Vector{FunnelConstraint}
    scaling::Any
    solution::FunnelSolution

    N::Int64  # number of subintervals (number of node - 1)
    w_funl::Float64  # weight for funnel cost
    w_vc::Float64  # weight for virtual control
    w_tr::Float64  # weight for trust-region
    tol_tr::Float64  # tolerance for trust-region
    tol_vc::Float64  # tolerance for virtual control
    tol_dyn::Float64  # tolerance for dynamics error
    max_iter::Int64  # maximum iteration
    verbosity::Bool

    flag_type::String
    funl_ctcs::Union{FunnelCTCS,Nothing}
    function FunnelRefinement(N::Int,max_iter::Int,
        dynamics::Dynamics,funl_dynamics::FunnelDynamics,funl_constraint::Vector{T},scaling::Scaling,
        w_funl::Float64,w_vc::Float64,w_tr::Float64,tol_tr::Float64,tol_vc::Float64,tol_dyn::Float64,
        verbosity::Bool;flag_type::String="Linear",funl_ctcs::Any=nothing) where T <: FunnelConstraint
        ix = dynamics.ix
        iu = dynamics.iu
        iq = funl_dynamics.iq
        iy = funl_dynamics.iy
        solution = FunnelSolution(N,ix,iu,iq,iy)
        if funl_ctcs === nothing
            println(flag_type," funnel and CTCS is ignored")
        else
            println(flag_type," funnel and CTCS is considered after first iteration")
        end
        new(dynamics,funl_dynamics,funl_constraint,scaling,solution,
            N,w_funl,w_vc,w_tr,tol_tr,tol_vc,tol_dyn,max_iter,verbosity,
            flag_type,funl_ctcs)
    end
end

function sdpopt!(fs::FunnelRefinement,xnom::Matrix,unom::Matrix,solver::String,iteration::Int64)
    N = fs.N
    ix = fs.dynamics.ix
    iu = fs.dynamics.iu
    iq = fs.funl_dynamics.iq
    is =  fs.funl_ctcs === nothing ? 0 : fs.funl_ctcs.is

    Qbar = fs.solution.Q
    Kbar = fs.solution.K
    Zbar = fs.solution.Z

    Sx = fs.scaling.Sx
    iSx = fs.scaling.iSx
    if solver == "Mosek"
        model = Model(Mosek.Optimizer)
        # set_optimizer_attribute(model, "MSK_IPAR_LOG", 0) # Turn off verbosity for Mosek
    elseif solver == "Clarabel"
        model = Model(Clarabel.Optimizer)
    else
        println("You should select Mosek or Clarabel")
    end

    # cvx variables (scaled)
    Qcvx = []
    Zcvx = []
    for i in 1:(N+1)
        push!(Qcvx, @variable(model, [1:ix, 1:ix], PSD))
        push!(Zcvx, @variable(model, [1:ix, 1:ix]))
    end
    @variable(model, bcvx[1:N+1])
    θ = fs.solution.θ # parameter

    # Q is PD
    very_small = 1e-4
    for i in 1:N+1
        @constraint(model, Sx*Qcvx[i]*Sx >= very_small * I(ix), PSDCone())
    end

    # scale reference trajectory
    Qbar_scaled = zeros(ix,ix,N+1)
    Zbar_scaled = zeros(ix,ix,N+1)
    for i in 1:N+1
        Qbar_scaled[:,:,i] .= iSx*Qbar[:,:,i]*iSx
        Zbar_scaled[:,:,i] .= iSx*Zbar[:,:,i]*iSx
    end

    # boundary condition
    boundary_initial!(fs,model,Sx*Qcvx[1]*Sx)
    boundary_final!(fs,model,Sx*Qcvx[end]*Sx)

    for i in 1:N+1
        Qi = Sx*Qcvx[i]*Sx
        Ki = Kbar[:,:,i]
        Yi = Ki*Qi
        Zi = Sx*Zcvx[i]*Sx
        bi = bcvx[i]
        if i<= N
            Qip = Sx*Qcvx[i+1]*Sx
            Kip = Kbar[:,:,i+1]
            Yip = Kip*Qip
            Zip = Sx*Zcvx[i+1]*Sx
            bip = bcvx[i+1]
        end
        if i <= N
            @constraint(model, vec(Qip) == fs.solution.Aq[:,:,i]*vec(Qi) +
                fs.solution.Bzm[:,:,i]*vec(Zi) + fs.solution.Bzp[:,:,i]*vec(Zip))
        end

        # Lyapunov condition
        if fs.flag_type == "Linear"
            @constraint(model, 0 <= Zcvx[i], PSDCone())
        elseif fs.flag_type == "Lsmooth"
            LMILsmooth!(fs,model,Qi,Yi,Zi,bi,i,iteration)
        end

        # constraints
        state_input_constraints!(fs,model::Model,Qi,Yi,xnom[:,i],unom[:,i])
    end

    # cost
    @variable(model, log_det_Q)
    @constraint(model, [log_det_Q; 1; vec(Sx*Qcvx[1]*Sx)] in MOI.LogDetConeSquare(ix))
    # cost_funl = - tr(Sx*Qcvx[1]*Sx) + tr(Sx*Qcvx[end]*Sx)
    # cost_funl = tr(Sx*Qcvx[end]*Sx)
    # cost_funl = - log_det_Q
    cost_funl = - tr(Sx*Qcvx[1]*Sx)

    # virtual control
    # for i in 1:N
    #     @constraint(model, [vc_t[i]; vc[:,i]] in MOI.NormOneCone(1 + is))
    # end
    # cost_vc = sum([vc_t[i] for i in 1:N])
    cost_vc = 0

    # trust region
    cost_tr = sum([dot(vec(Qcvx[i]-Qbar_scaled[:,:,i]),vec(Qcvx[i]-Qbar_scaled[:,:,i])) +
        dot(vec(Zcvx[i]-Zbar_scaled[:,:,i]),vec(Zcvx[i]-Zbar_scaled[:,:,i]))
        for i in 1:N+1])
    w_tr = iteration > 1 ? fs.w_tr : 0

    cost_all = fs.w_funl * cost_funl + fs.w_vc * cost_vc + w_tr * cost_tr
   
    @objective(model,Min,cost_all)
    optimize!(model)

    for i in 1:N+1
        fs.solution.Q[:,:,i] .= Sx*value.(Qcvx[i])*Sx
        fs.solution.Z[:,:,i] .= Sx*value.(Zcvx[i])*Sx
    end
    fs.solution.b = value.(bcvx)

    return value(cost_all),value(cost_funl),value(cost_vc),value(cost_tr)
end

function run(fs::FunnelRefinement,Q0::Array{Float64,3},K0::Array{Float64,3},Z0::Array{Float64,3},Qi::Matrix,Qf::Matrix,xnom::Matrix,unom::Matrix,dtnom::Vector,solver::String,θ0=nothing)
    fs.solution.Q .= Q0
    fs.solution.K .= K0
    fs.solution.Z .= Z0

    fs.solution.Qi .= Qi 
    fs.solution.Qf .= Qf

    if θ0 !== nothing
        fs.solution.θ = θ0
    end

    for iteration in 1:fs.max_iter
        # discretization & linearization
        if fs.funl_ctcs === nothing
            fs.solution.Aq,fs.solution.Bzm,fs.solution.Bzp = discretize_foh(fs.funl_dynamics,
                fs.dynamics,xnom[:,1:N],unom,dtnom,fs.solution.Q[:,:,1:N],fs.solution.K,fs.solution.Z)
        # else
        #     (
        #         fs.solution.Aq,fs.solution.Bym,fs.solution.Byp,fs.solution.Bzm,fs.solution.Bzp,
        #         fs.solution.Sq,fs.solution.Sym,fs.solution.Syp,fs.solution.Szm,fs.solution.Szp,fs.solution.SZ,
        #         ~,~,s_prop 
        #     ) = discretize_foh(fs.funl_dynamics,fs.dynamics,fs.funl_ctcs,
        #         xnom[:,1:N],unom,dtnom,fs.solution.Q[:,:,1:N],fs.solution.Y,fs.solution.Z)
        #     println(s_prop)
        end

        # solve subproblem
        c_all, c_funl, c_vc, c_tr = sdpopt!(fs,xnom,unom,solver,iteration)

        # propagate
        Qfwd,fs.solution.tprop,fs.solution.xprop,fs.solution.uprop,fs.solution.Qprop,fs.solution.Kprop =  propagate_multiple_FOH(fs.funl_dynamics,fs.dynamics,
            xnom,unom,dtnom,fs.solution.Q,fs.solution.K,fs.solution.Z,false)
        dyn_error = maximum([norm(Qfwd[:,:,i] - fs.solution.Q[:,:,i],2) for i in 1:N+1])

        if fs.verbosity == true && iteration == 1
            println("+--------------------------------------------------------------------------------------------------+")
            println("|                                   ..:: Penalized Trust Region ::..                               |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
            println("| iter. |    cost    |    tof    |   funl    |   rate    |  param  | log(vc) | log(tr)  | log(dyn) |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
        end
        @printf("|%-2d     |%-7.2f     |%-7.3f   |%-7.3f    |%-7.3f    |%-5.3f    |%-5.1f    | %-5.1f    |%-5.1e   |\n",
            iteration,
            c_all,-1,c_funl,-1,
            -1,
            log10(abs(c_vc)), log10(abs(c_tr)), log10(abs(dyn_error)))

        flag_vc::Bool = c_vc < fs.tol_vc
        flag_tr::Bool = c_tr < fs.tol_tr
        flag_dyn::Bool = dyn_error < fs.tol_dyn

        if flag_vc && flag_tr && flag_dyn
            println("+--------------------------------------------------------------------------------------------------+")
            println("Converged!")
            break
        end
    end
end