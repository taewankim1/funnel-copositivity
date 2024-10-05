using LinearAlgebra
using Printf
using JuMP
using MosekTools
using Clarabel

include("funl_dynamics.jl")
include("funl_utils.jl")
include("funl_constraint.jl")
include("funl_ctcs.jl")
include("../trajopt/dynamics.jl")
include("../trajopt/scaling.jl")

mutable struct FunnelSolution
    Q::Array{Float64,3}
    # K::Array{Float64,3}
    Y::Array{Float64,3}
    lam::Array{Float64,2}

    gamma::Vector{Float64}
    beta::Vector{Float64}

    Qi::Matrix{Float64}
    Qf::Matrix{Float64}

    lambda_w::Float64
    t::Vector{Float64}
    function FunnelSolution(N::Int64,ix::Int64,iu::Int64,lambda_w::Float64)
        Q = zeros(ix,ix,N+1)
        # K = zeros(iu,ix,N+1)
        Y = zeros(iu,ix,N+1)
        lam = zeros(2,N+1)
        gamma = ones(N)
        beta = ones(N)

        Qi = zeros(ix,ix)
        Qf = zeros(ix,ix)
        
        new(Q,Y,lam,gamma,beta,Qi,Qf,lambda_w)
    end
end

struct FunnelSynthesis
    dynamics::Dynamics
    funl_constraint::Vector{FunnelConstraint}
    scaling::Any
    solution::FunnelSolution

    N::Int64  # number of subintervals (number of node - 1)
    flag_copositivity_type::Int

    verbosity::Bool
    function FunnelSynthesis(N::Int,
        lambda_w::Float64,
        dynamics::Dynamics,
        funl_constraint::Vector{T},scaling::Scaling;
        verbosity::Bool=true,flag_copositivity_type::Int=1) where T <: FunnelConstraint
        ix = dynamics.ix
        iu = dynamics.iu
        solution = FunnelSolution(N,ix,iu,lambda_w)
        new(dynamics,funl_constraint,scaling,solution,N,flag_copositivity_type,verbosity)
    end
end

function get_block_LMI(fs,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
    if typeof(fs.funl_dynamics) == LinearFOH || typeof(fs.funl_dynamics) == LinearSOH
        Ai,Bi = diff(fs.dynamics,xi,ui)
        Aj,Bj = diff(fs.dynamics,xj,uj)
        Wij = Ai*Qj + Qj'*Ai' + Bi*Yj + Yj'*Bi' + 0.5*fs.funl_dynamics.alpha*(Qj+Qj') - Z
        Wji = Aj*Qi + Qi'*Aj' + Bj*Yi + Yi'*Bj' + 0.5*fs.funl_dynamics.alpha*(Qi+Qi') - Z
    elseif typeof(fs.funl_dynamics) == LinearDLMI
        Wij = - Z
        Wji = - Z
    end
    LMI11 = Wij + Wji
    if fs.flag_type == "Linear"
        return LMI11
    end
    θ = fs.solution.θ
    iμ = fs.dynamics.iμ
    iψ = fs.dynamics.iψ
    N11 = diagm(θ ./ ( fs.dynamics.β .* fs.dynamics.β))
    N22i =  bi * θ .* Matrix{Float64}(I, iψ, iψ)
    N22j =  bj * θ .* Matrix{Float64}(I, iψ, iψ)
    LMI21 = (N22i+N22j) * fs.dynamics.G'
    LMI22 = -(N22i+N22j)
    LMI31i = fs.dynamics.Cμ * Qi + fs.dynamics.Dμu * Yi
    LMI31j = fs.dynamics.Cμ * Qj + fs.dynamics.Dμu * Yj
    LMI31 = LMI31i + LMI31j
    LMI32 = 2*zeros(iμ,iψ)
    LMI33 = -2*N11
    LMI = 0.5 * [LMI11 LMI21' LMI31';
        LMI21 LMI22 LMI32';
        LMI31 LMI32 LMI33
    ]
    return LMI
end

function stack_LMI(LMI11,LMI21,LMI31,LMI41,
                    LMI22,LMI32,LMI42,
                            LMI33,LMI43,
                                LMI44)
    row1 = [LMI11 LMI21' LMI31' LMI41']
    row2 = [LMI21 LMI22 LMI32' LMI42']
    row3 = [LMI31 LMI32 LMI33 LMI43']
    row4 = [LMI41 LMI42 LMI43 LMI44]
    LMI = [row1;row2;row3;row4]
    return LMI
end


function get_b_LMI(fs,Q,Y,b)
    tmp12 = fs.dynamics.Cv*Q + fs.dynamics.Dvu*Y
    Bound_b = [b * I(fs.dynamics.iv) tmp12;
        tmp12' Q
    ]
    return Bound_b
end

function block_LMIs!(fs,model::Model,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
    LMI = get_block_LMI(fs,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
    @constraint(model, LMI <= 0, PSDCone())
    return LMI
end

function boundary_initial!(fs,model::Model,Q1)
    @constraint(model, Q1 >= fs.solution.Qi, PSDCone())
end

function boundary_final!(fs,model::Model,Qend)
    # diag([0.2**2,0.2**2,np.deg2rad(10)**2])
    Qf = diagm([0.2^2,0.2^2,deg2rad(10^2)])
    @constraint(model, Qend <= Qf, PSDCone())
end

function state_input_constraints!(fs,model::Model,Qi,Yi,xnom,unom,idx)
    N_constraint = size(fs.funl_constraint,1)
    for j in 1:N_constraint
        impose!(fs.funl_constraint[j],model,Qi,Yi,xnom,unom,idx)
    end
end

function sdpopt!(fs::FunnelSynthesis,xnom::Matrix,unom::Matrix,tnom::Vector,
    Qmax::Array{Float64,3},Rmax::Array{Float64,3},solver::String)

    N = fs.N
    ix = fs.dynamics.ix
    iu = fs.dynamics.iu
    iq = fs.dynamics.iq
    iphi = fs.dynamics.iphi
    ir = fs.dynamics.ir
    ip = fs.dynamics.ip

    Sx = fs.scaling.Sx
    iSx = fs.scaling.iSx
    Su = fs.scaling.Su
    iSu = fs.scaling.iSu

    lambda_w = fs.solution.lambda_w

    if solver == "Mosek"
        model = Model(Mosek.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_LOG", fs.verbosity) # verbosity for Mosek
        # set_optimizer_attribute(model, "MSK_IPAR_INTPNT_SCALING", Mosek.MSK_ON)
        # set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_PFEAS", 1e-6)
        # set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_DFEAS", 1e-6)
        # set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-7)
        # set_optimizer_attribute(model, "MSK_IPAR_INTPNT_MAX_ITERATIONS", 1000)
        # set_optimizer_attribute(model, "MSK_IPAR_LOG", 10)
        # set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-3)
        # set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", 600.0)
    elseif solver == "Clarabel"
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", true) # verbosity for Mosek
    else
        println("You should select Mosek or Clarabel")
    end

    # cvx variables (scaled)
    Qcvx = []
    Ycvx = []
    for i in 1:(N+1)
        push!(Qcvx, @variable(model, [1:ix, 1:ix], PSD))
        push!(Ycvx, @variable(model, [1:iu, 1:ix]))
    end
    @variable(model, lamcvx[1:2,1:N+1] .>= 0)

    # alias gamma and beta 
    beta = fs.solution.beta
    gamma = fs.solution.gamma

    # Q is PD
    very_small = 1e-6
    for i in 1:N+1
        @constraint(model, Sx*Qcvx[i]*Sx >= very_small .* Matrix(1.0I,ix,ix), PSDCone())
    end

    # # boundary condition
    # boundary_initial!(fs,model,Sx*Qcvx[1]*Sx)
    # boundary_final!(fs,model,Sx*Qcvx[end]*Sx)

    function get_N1(lam::Vector)::Matrix
        N11 = [lam[1]*1.0I(ix+iu+iw) zeros(ix+iu+iw,iq)]
        N22 = [zeros(iq,ix+iu+iw) lam[2]*1.0I(iq)]
        N1 = [N11;N22]
        return N1
    end

    function get_N2(lam::Vector,gamma_sq::Float64,beta_sq::Float64)::Matrix
        N11 = [lam[1]*beta_sq*1.0I(ix) zeros(ix,iphi)]
        N22 = [zeros(iphi,ix) lam[2]*gamma_sq*1.0I(iphi)]
        N2 = [N11;N22]
        return N2
    end

    function get_H(Qi,Qj,Yi,Yj,lami,lamj,Ai,Aj,Bi,Bj,Fi,Fj,dQ,gamma_sq,beta_sq) 
        Wij = Ai*Qj + Bi*Yj + Qj*Ai' + Yj'*Bi' + lambda_w * Qj
        Wji = Aj*Qi + Bj*Yi + Qi*Aj' + Yi'*Bj' + lambda_w * Qi

        Li = fs.dynamics.C*Qi + fs.dynamics.D*Yi
        Lj = fs.dynamics.C*Qj + fs.dynamics.D*Yj
        
        LMI11 = Wij + Wji - 2*dQ
        LMI21 = (get_N2(lami,gamma_sq,beta_sq) + get_N2(lamj,gamma_sq,beta_sq)) * fs.dynamics.E'
        LMI31 = Fi' + Fj'
        LMI41 = Li + Lj
        LMI22 = - (get_N2(lami,gamma_sq,beta_sq)+get_N2(lamj,gamma_sq,beta_sq))
        LMI32 = zeros(iw,ip)
        LMI42 = zeros(ir,ip)
        LMI33 = -2*lambda_w * 1.0I(iw)
        LMI43 = 2*fs.dynamics.G
        LMI44 = - (get_N1(lami) + get_N1(lamj))
        H_ij = - 1/2 * stack_LMI(LMI11,LMI21,LMI31,LMI41,
                                        LMI22,LMI32,LMI42,
                                        LMI33,LMI43,
                                        LMI44)
        return H_ij
    end

    for i in 1:N+1
        Qi = Sx*Qcvx[i]*Sx
        Yi = Su*Ycvx[i]*Sx
        lami = lamcvx[:,i]
        xi = xnom[:,i]
        ui = unom[:,i]
        Ai,Bi,Fi = diff_ABF(fs.dynamics,xi,ui)
        if i <= N
            Qj = Sx*Qcvx[i+1]*Sx
            Yj = Su*Ycvx[i+1]*Sx
            lamj = lamcvx[:,i+1]
            xj = xnom[:,i+1]
            uj = unom[:,i+1]
            Aj,Bj,Fj = diff_ABF(fs.dynamics,xj,uj)
            delt = tnom[i+1] - tnom[i]
            dQ = (Qj-Qi) / delt
            if fs.flag_copositivity_type == 1
                gamma_sq = fs.solution.gamma[i]^2
                beta_sq = fs.solution.beta[i]^2

                H_ii = get_H(Qi,Qi,Yi,Yi,lami,lami,Ai,Ai,Bi,Bi,Fi,Fi,dQ,gamma_sq,beta_sq)
                H_ij = get_H(Qi,Qj,Yi,Yj,lami,lamj,Ai,Aj,Bi,Bj,Fi,Fj,dQ,gamma_sq,beta_sq)
                H_jj = get_H(Qj,Qj,Yj,Yj,lamj,lamj,Aj,Aj,Bj,Bj,Fj,Fj,dQ,gamma_sq,beta_sq)

                @constraint(model,H_ii >= 0, PSDCone())
                @constraint(model,H_ij >= 0, PSDCone())
                @constraint(model,H_jj >= 0, PSDCone())
            elseif fs.flag_copositivity_type == 2
                error("Copositivity type 2 not implemented yet")
            else
                error("Copositivity type must be 1 or 2")
            end
        end

        # constraints
        state_input_constraints!(fs,model,Qi,Yi,xi,ui,i)

        # constraints on Qmax and Rmax
        @constraint(model, Qi <= Qmax[:,:,i], PSDCone())
        @constraint(model, [Qi Yi';Yi Rmax[:,:,i]] >= 0, PSDCone())
    end

    # cost
    # @variable(model, log_det_Q)
    # @constraint(model, [log_det_Q; 1; vec(Sx*Qcvx[1]*Sx)] in MOI.LogDetConeSquare(ix))
    cost_funl = - tr(Sx*Qcvx[1]*Sx) + tr(Sx*Qcvx[end]*Sx)
    # cost_funl = tr(Sx*Qcvx[end]*Sx)
    # cost_funl = - log_det_Q # not stable
    # cost_funl = - tr(Sx*Qcvx[1]*Sx)

    @objective(model,Min,cost_funl)
    optimize!(model)

    for i in 1:N+1
        fs.solution.Q[:,:,i] .= Sx*value.(Qcvx[i])*Sx
        fs.solution.Y[:,:,i] .= Su*value.(Ycvx[i])*Sx
    end
    fs.solution.lam = value.(lamcvx)
    return value(cost_funl)
end

function run!(fs::FunnelSynthesis,
        # Qi::Matrix,Qf::Matrix,
        gamma::Vector,beta::Vector,
        xnom::Matrix,unom::Matrix,tnom::Vector,
        Qmax::Array{Float64,3},Rmax::Array{Float64,3},
        solver::String)
    # fs.solution.Qi .= Qi 
    # fs.solution.Qf .= Qf

    fs.solution.gamma .= gamma
    fs.solution.beta .= beta

    # solve subproblem
    c_all = sdpopt!(fs,xnom,unom,tnom,Qmax,Rmax,solver)
    return c_all

    # # propagate
    # (
    #     Qfwd,
    #     fs.solution.tprop,fs.solution.xprop,fs.solution.uprop,
    #     fs.solution.Qprop,fs.solution.Yprop,fs.solution.Zprop
    #     ) =  propagate_multiple_FOH(fs.funl_dynamics,fs.dynamics,
    #     xnom,unom,dtnom,fs.solution.Q,fs.solution.Y,fs.solution.Z,false)
    # dyn_error = maximum([norm(Qfwd[:,:,i] - fs.solution.Q[:,:,i],2) for i in 1:N+1])

    # if fs.verbosity == true && iteration == 1
    #     println("+--------------------------------------------------------------------------------------------------+")
    #     println("|                                   ..:: Penalized Trust Region ::..                               |")
    #     println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
    #     println("| iter. |    cost    |    tof    |   funl    |   rate    |  param  | log(vc) | log(tr)  | log(dyn) |")
    #     println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
    # end
    # @printf("|%-2d     |%-7.2f     |%-7.3f   |%-7.3f    |%-7.3f    |%-5.3f    |%-5.1f    | %-5.1f    |%-5.1e   |\n",
    #     iteration,
    #     c_all,-1,c_funl,-1,
    #     -1,
    #     log10(abs(c_vc)), log10(abs(c_tr)), log10(abs(dyn_error)))
end