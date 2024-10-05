
using LinearAlgebra
using JuMP
using MosekTools

abstract type FunnelConstraint end

struct StateConstraint <: FunnelConstraint
    a::Vector
    b::Float64
    function StateConstraint(a::Vector,b::Float64)
        new(a,b)
    end
end

struct InputConstraint <: FunnelConstraint
    a::Vector
    b::Float64
    function InputConstraint(a::Vector,b::Float64)
        new(a,b)
    end
end

function impose!(constraint::StateConstraint,model::Model,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector,idx::Int)
    a = constraint.a
    b = constraint.b

    LMI = [(b-a'*xnom)*(b-a'*xnom) a'*Q;
        Q*a Q
    ]
    @constraint(model, 0 <= LMI, PSDCone())
end

function impose!(constraint::InputConstraint,model::Model,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector,idx::Int)
    a = constraint.a
    b = constraint.b

    LMI = [(b-a'*unom)*(b-a'*unom) a'*Y;
        Y'*a Q
    ]
    @constraint(model, 0 <= LMI, PSDCone())
end

struct ObstacleAvoidance <: FunnelConstraint
    H::Matrix
    c::Vector
    function ObstacleAvoidance(H::Matrix,c::Vector)
        new(H,c)
    end
end

function impose!(constraint::ObstacleAvoidance,model::Model,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector,idx::Int)
    H = constraint.H
    c = constraint.c
    M = [1 0 0;0 1 0]
    a = - M'*H'*H*(M*xnom-c) / norm(H*(M*xnom-c))
    s = 1 - norm(H*(M*xnom-c))
    b = -s + a'*xnom

    LMI = [(b-a'*xnom)*(b-a'*xnom) a'*Q;
        Q*a Q
    ]
    @constraint(model, 0 <= LMI, PSDCone())
end

struct WayPoint <: FunnelConstraint
    Qpos_max::Matrix{Float64}
    function WayPoint(Qmax::Matrix)
        new(Qmax)
    end
end

function impose!(constraint::WayPoint,model::Model,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector,idx::Int)
    # hard coding
    # 1,4,7,10,13,16
    if (idx == 4) || (idx == 7) || (idx == 10) || (idx == 13)
        @constraint(model, Q[1:3,1:3] <= constraint.Qpos_max, PSDCone())
    end
end

function get_Rmax_unicycle(unom::Matrix,N::Int,vmax::Float64,vmin::Float64,wmax::Float64,wmin::Float64)
    iu = 2
    Rmax = zeros(iu,iu,N+1)
    for idx = 1:N+1
    u = unom[:,idx]
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_IPAR_LOG", 0) # Turn off verbosity for Mosek
    Rcvx = @variable(model,[1:iu,1:iu], PSD)

    @constraint(model, [vmax-u'*[1;0];Rcvx*[1;0]] in SecondOrderCone())
    @constraint(model, [-vmin+u'*[1;0];Rcvx*[1;0]] in SecondOrderCone())
    @constraint(model, [wmax-u'*[0;1];Rcvx*[0;1]] in SecondOrderCone())
    @constraint(model, [-wmin+u'*[0;1];Rcvx*[0;1]] in SecondOrderCone())

    @variable(model, log_det_Q)
    @constraint(model, [log_det_Q; 1; vec(Rcvx)] in MOI.LogDetConeSquare(iu))
    cost = - log_det_Q # not stable

    @objective(model,Min,cost)
    optimize!(model)

    Rmax[:,:,idx] .= value.(Rcvx) * value.(Rcvx)
    end
    return Rmax
end