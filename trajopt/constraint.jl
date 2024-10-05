using LinearAlgebra
using JuMP

abstract type Constraint end

struct InputLinear <: Constraint
    A::Matrix
    b::Vector
    function InputLinear(A::Matrix,b::Vector)
        new(A,b)
    end
end

function impose!(constraint::InputLinear,model::Model,x::Vector,u::Vector,xbar::Vector=[nothing],ubar::Vector=[nothing];idx::Int=0)
    A = constraint.A
    b = constraint.b
    # return A * u == b
    @constraint(model, A*u .<= b) 
end

struct Obstacle <: Constraint
    H::Matrix
    c::Vector
    function Obstacle(H::Matrix,c::Vector)
        new(H,c)
    end
end

function impose!(constraint::Obstacle,model::Model,x::Vector,u::Vector,xbar::Vector,ubar::Vector=[nothing];idx::Int=0)
    # ||H(r-c)|| >= 1

    # obstacle parameters
    H = constraint.H
    c = constraint.c

    # position in state
    r = x[1:2]
    rbar = xbar[1:2]
    
    norm_H_rbar_c = norm(H*(rbar-c)) # ||H(rbar-c)||
    sbar = 1 - norm_H_rbar_c
    dsbar = - H'*H*(rbar-c) / norm_H_rbar_c

    A = dsbar'
    b = -sbar + dsbar' * rbar
    @constraint(model, A*r .<= b) 
end

struct PDG <: Constraint
    m_dry::Float64

    vmax::Float64
    wmax::Float64

    gamma_s::Float64
    theta_max::Float64

    Fmin::Float64
    Fmax::Float64
    tau_max::Float64
    delta_max::Float64
    function PDG()
        m_dry = 750

        vmax = 90
        wmax = deg2rad(5)
        glide_slope_max = deg2rad(20) 
        theta_max = deg2rad(90)

        Fmin = 600
        Fmax = 3000
        tau_max = 50
        delta_max = deg2rad(20)

        new(m_dry,vmax,wmax,glide_slope_max,theta_max,Fmin,Fmax,tau_max,delta_max)
    end
end

function impose!(pdg::PDG,model::Model,x::Vector,u::Vector,xbar::Vector=[nothing],ubar::Vector=[nothing];idx::Int=0)
    # m rx ry rz vx vy vz roll pitch yaw wx wy wz
    # 1  2  3  4  5  6  7 . 8 . 9    10  11 12 13
    # mass
    m = x[1]
    @constraint(model,pdg.m_dry <= m)

    # maximum velocity
    v = x[5:7]
    @constraint(model,[pdg.vmax;v] in SecondOrderCone())

    # maximum angular velocity
    w = x[11:13]
    @constraint(model,[pdg.wmax;w] in SecondOrderCone())

    # glide slope angle
    @constraint(model, [x[4]/tan(pdg.gamma_s); x[2:3]] in SecondOrderCone())

    # maximum tilt
    roll = x[8]
    pitch = x[9]
    @constraint(model, [roll,-roll] .<= [pdg.theta_max;pdg.theta_max])
    @constraint(model, [pitch,-pitch] .<= [pdg.theta_max;pdg.theta_max])

    # minimum thrust (non convex)
    F = u[1:3]
    Fbar = ubar[1:3]
    @constraint(model, pdg.Fmin - Fbar'*F / norm(Fbar,2) <= 0)

    # maximum thrust
    @constraint(model, [pdg.Fmax;F] in SecondOrderCone())

    # gimbal angle
    @constraint(model, [u[3]/cos(pdg.delta_max);F] in SecondOrderCone())

    # maximum torque
    T = u[4:6]
    # @constraint(model, [pdg.tau_max;T] in SecondOrderCone())
    @constraint(model, [pdg.tau_max; T] in MOI.NormInfinityCone(1 + length(T)))
end

function initial_condition!(dynamics::Dynamics,model::Model,x1::Vector,xi::Vector)
    @constraint(model,x1 == xi)
end
function final_condition!(dynamics::Dynamics,model::Model,xN::Vector,xf::Vector;uN::Vector)
    @constraint(model,xN == xf)
end
# function final_condition!(dynamics::Rocket,model::Model,xN::Vector,xf::Vector;uN::Vector)
#     @constraint(model,xN[2:dynamics.ix] == xf[2:dynamics.ix])
# end

# struct ThreeDOFManipulatorConstraint <: Constraint
#     tau_max::Float64
#     dq_max::Float64
#     function ThreeDOFManipulatorConstraint(tau_max::Float64,dq_max::Float64)
#         new(tau_max,dq_max)
#     end
# end

# function final_condition!(dynamics::ThreeDOFManipulatorDynamics,model::Model,xN::Vector,xf::Vector;uN::Vector=nothing)
#     @constraint(model,xN == xf)
#     # if uN !== nothing
#     #     @constraint(model,uN == xf)
#     # end
# end

# function impose!(constraint::ThreeDOFManipulatorConstraint,model::Model,x::Vector,u::Vector,xbar::Vector=[nothing],ubar::Vector=[nothing];idx::Int=0)
#     q1 = x[1]
#     q2 = x[2]
#     q3 = x[3]
#     q1bar = xbar[1]
#     q2bar = xbar[2]
#     q3bar = xbar[3]

#     dq1 = x[4]
#     dq2 = x[5]
#     dq3 = x[6]
#     dq1bar = xbar[4]
#     dq2bar = xbar[5]
#     dq3bar = xbar[6]

#     tau1 = u[1]
#     tau2 = u[2]
#     tau3 = u[3]

#     @constraint(model, q1 <= pi)
#     @constraint(model, q2 <= pi)
#     @constraint(model, q3 <= pi)
#     @constraint(model, - q1 <= pi)
#     @constraint(model, - q2 <= pi)
#     @constraint(model,  - q3 <= pi)

#     @constraint(model, dq1 <= constraint.dq_max)
#     @constraint(model, dq2 <= constraint.dq_max)
#     @constraint(model, dq3 <= constraint.dq_max)
#     @constraint(model, - dq1 <= constraint.dq_max)
#     @constraint(model, - dq2 <= constraint.dq_max)
#     @constraint(model,  - dq3 <= constraint.dq_max)

#     @constraint(model, tau1 <= constraint.tau_max)
#     @constraint(model, tau2 <= constraint.tau_max)
#     @constraint(model, tau3 <= constraint.tau_max)
#     @constraint(model, - tau1 <= constraint.tau_max)
#     @constraint(model, - tau2 <= constraint.tau_max)
#     @constraint(model, - tau3 <= constraint.tau_max)
# end

# struct ThreeDOFManipulatorMultiphaseConstraint <: Constraint
#     N1::Int64
#     N2::Int64
#     l1::Float64
#     l2::Float64
#     l3::Float64
#     function ThreeDOFManipulatorMultiphaseConstraint(N::Int,l1::Float64,l2::Float64,l3::Float64)
#         N1 = Int(ceil(N*0.5))
#         N2 = N - N1
#         new(N1,N2,l1,l2,l3)
#     end
# end

# function impose!(constraint::ThreeDOFManipulatorMultiphaseConstraint,model::Model,x::Vector,u::Vector,xbar::Vector=[nothing],ubar::Vector=[nothing];idx::Int=0)
#     xf = zeros(3)
#     xf[1] = - pi/3
#     xf[2] = 2 * pi / 3
#     xf[3] = - pi / 3
#     # x3bar
#     l1 = constraint.l1
#     l2 = constraint.l2
#     l3 = constraint.l3
#     q1_bar = xbar[1]
#     q2_bar = xbar[2]
#     q3_bar = xbar[3]

#     x3_bar = l1 * cos(q1_bar) + l2 * cos(q1_bar+q2_bar) + l3 * cos(q1_bar+q2_bar+q3_bar)
#     y3_bar = l1 * sin(q1_bar) + l2 * sin(q1_bar+q2_bar) + l3 * sin(q1_bar+q2_bar+q3_bar)

#     J = zeros(2,3)
#     J[1,1] = -l1*sin(q1_bar) - l2*sin(q1_bar + q2_bar) - l3*sin(q1_bar + q2_bar + q3_bar)
#     J[1,2] = -l2*sin(q1_bar + q2_bar) - l3*sin(q1_bar + q2_bar + q3_bar)
#     J[1,3] = -l3*sin(q1_bar + q2_bar + q3_bar)
#     J[2,1] = l1*cos(q1_bar) + l2*cos(q1_bar + q2_bar) + l3*cos(q1_bar + q2_bar + q3_bar)
#     J[2,2] = l2*cos(q1_bar + q2_bar) + l3*cos(q1_bar + q2_bar + q3_bar)
#     J[2,3] = l3*cos(q1_bar + q2_bar + q3_bar)

#     if (idx == constraint.N1 + 1)
#         @constraint(model,x[1:3] == xf)
#         @constraint(model,x[4:6] == zeros(3))
#     end
#     if (idx < constraint.N1 + 1)
#         @constraint(model, x3_bar .+ J[1,:]' * (x[1:3].-xbar[1:3]) <= 2.0)
#     end
#     if (idx > constraint.N1 + 1)
#         @constraint(model, 2 * x[1] == - x[2])
#         # @constraint(model, y3_bar .+ J[2,:]' * (x[1:3].-xbar[1:3]) == 0.0)
#         # @constraint(model, y3_bar .+ J[2,:]' * (x[1:3].-xbar[1:3]) >= - 0.1)
#         @constraint(model,sum(x[1:3]) == 0.0)
#     end

# end

struct QuadrotorConstraint <: Constraint
    v_max::Float64
    att_max::Float64
    att_vel_max::Float64
    F_max::Float64
    M_max::Float64
end

# function final_condition!(dynamics::QuadrotorDynamics,model::Model,xN::Vector,xf::Vector;uN::Vector=nothing)
#     @constraint(model,xN == xf)
#     # if uN !== nothing
#     #     @constraint(model,uN == xf)
#     # end
# end

function impose!(constraint::QuadrotorConstraint,model::Model,x::Vector,u::Vector,xbar::Vector=[nothing],ubar::Vector=[nothing];idx::Int=0)
    # maximum velocity
    v = x[4:6]
    @constraint(model,[constraint.v_max;v] in SecondOrderCone())

    # attitude
    roll = x[7]
    pitch = x[8]
    @constraint(model, [roll,-roll] .<= [constraint.att_max;constraint.att_max])
    @constraint(model, [pitch,-pitch] .<= [constraint.att_max;constraint.att_max])

    # angular velocity
    w = x[10:12]
    @constraint(model,[constraint.att_vel_max;w] in SecondOrderCone())

    # Force and moments
    Fz = u[1]
    Mx = u[2]
    My = u[3]
    Mz = u[4]
    @constraint(model, Fz >= 0)
    @constraint(model, Fz <= constraint.F_max)
    @constraint(model, [Mx,-Mx] .<= [constraint.M_max;constraint.M_max])
    @constraint(model, [My,-My] .<= [constraint.M_max;constraint.M_max])
    @constraint(model, [Mz,-Mz] .<= [constraint.M_max;constraint.M_max])
end

struct QuadrotorMultiphaseConstraint <: Constraint
    N1::Int64
    N2::Int64
    N3::Int64
    N4::Int64
    N5::Int64
    N6::Int64
    P1::Vector{Float64}
    P2::Vector{Float64}
    P3::Vector{Float64}
    P4::Vector{Float64}
    P5::Vector{Float64}
    h::Float64
    function QuadrotorMultiphaseConstraint(N::Int64,r::Float64,h::Float64)
        @assert(N % 5 == 0)
        deviation = div(N,5)
        N1 = 1
        N2 = N1 + deviation
        N3 = N2 + deviation
        N4 = N3 + deviation
        N5 = N4 + deviation
        N6 = N5 + deviation
        @assert(N6 == N+1)

        angle = 2*pi / 5
        r = 5
        theta1 = - pi / 2 - angle / 2
        theta3 = theta1 + angle
        theta5 = theta3 + angle
        theta2 = theta5 + angle
        theta4 = theta2 + angle
        theta6 = theta4 + angle
        P1 = [r * cos(theta1), r * sin(theta1)]
        P2 = [r * cos(theta2), r * sin(theta2)]
        P3 = [r * cos(theta3), r * sin(theta3)]
        P4 = [r * cos(theta4), r * sin(theta4)]
        P5 = [r * cos(theta5), r * sin(theta5)]
        P6 = [r * cos(theta6), r * sin(theta6)]
        @assert(isapprox(P1,P6))
        new(N1,N2,N3,N4,N5,N6,P1,P2,P3,P4,P5,h)
    end
end

function impose!(constraint::QuadrotorMultiphaseConstraint,model::Model,x::Vector,u::Vector,xbar::Vector=[nothing],ubar::Vector=[nothing];idx::Int=0)
    if(idx == constraint.N1)
        @constraint(model,x[1:2] == constraint.P1)
        @constraint(model,x[3] == constraint.h)
    elseif(idx == constraint.N2)
        @constraint(model,x[1:2] == constraint.P2)
        @constraint(model,x[3] == constraint.h)
    elseif(idx == constraint.N3)
        @constraint(model,x[1:2] == constraint.P3)
        @constraint(model,x[3] == constraint.h)
    elseif(idx == constraint.N4)
        @constraint(model,x[1:2] == constraint.P4)
        @constraint(model,x[3] == constraint.h)
    elseif(idx == constraint.N5)
        @constraint(model,x[1:2] == constraint.P5)
        @constraint(model,x[3] == constraint.h)
    # elseif(idx == N6) # imposed by final boundary condition
    end
end
    