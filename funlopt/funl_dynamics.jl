include("../trajopt/dynamics.jl")
include("../trajopt/discretize.jl")
using LinearAlgebra
using BlockDiagonals
using SparseArrays

abstract type FunnelDynamics end

function com_mat(m, n)
    A = reshape(1:m*n, n, m)  # Note the swapped dimensions for Julia
    v = reshape(A', :)

    P = Matrix{Int}(I, m*n, m*n)  # Create identity matrix
    P = P[v, :]
    return P'
end

struct LinearFOH <: FunnelDynamics
    alpha::Float64 # decay rate
    ix::Int
    iu::Int

    iq::Int
    iy::Int
    ip::Int

    Cn::Matrix # commutation matrix
    Cm::Matrix # commutation matrix
    function LinearFOH(alpha,ix,iu)
        Cn = com_mat(ix,ix)
        Cm = com_mat(iu,ix)
        # ip = ix*ix# + 1 # Z11, b
        ip = 0
        new(alpha,ix,iu,ix*ix,ix*iu,ip,Cn,Cm)
    end
end

function forward(model::LinearFOH, q::Vector, y::Vector, z::Vector, A::Matrix, B::Matrix)
    Q = reshape(q,(model.ix,model.ix))
    Y = reshape(y,(model.iu,model.ix))
    Z = reshape(z,(model.ix,model.ix))
   
    dQ = Z
    return vec(dQ)
end

function diff(model::LinearFOH,A::Matrix,B::Matrix)
    Imat = sparse(I(model.ix))
    Aq = sparse(zeros(model.iq,model.iq))
    Bq = sparse(zeros(model.iq,model.iy))
    Sq = kron(Imat,Imat)
    return Aq,Bq,Sq
end

struct LinearSOH <: FunnelDynamics
    alpha::Float64 # decay rate
    ix::Int
    iu::Int

    iq::Int
    iy::Int
    ip::Int

    Cn::Matrix # commutation matrix
    Cm::Matrix # commutation matrix
    function LinearSOH(alpha,ix,iu)
        Cn = com_mat(ix,ix)
        Cm = com_mat(iu,ix)
        # ip = ix*ix# + 1 # Z11, b
        ip = 0
        new(alpha,ix,iu,ix*ix,ix*iu,ip,Cn,Cm)
    end
end

function forward(model::LinearSOH, q::Vector, y::Vector, z::Vector, A::Matrix, B::Matrix)
    Q = reshape(q,(model.ix,model.ix))
    Y = reshape(y,(model.iu,model.ix))
    Z = reshape(z,(model.ix,model.ix))
    dQ = Z
    return vec(dQ)
end

function diff(model::LinearSOH,A::Matrix,B::Matrix)
    Imat = I(model.ix)
    Aq = zeros(model.iq,model.iq)
    Bq = zeros(model.iq,model.iy)
    Sq = kron(Imat,Imat)
    return Aq,Bq,Sq
end
struct LinearDLMI <: FunnelDynamics
    alpha::Float64 # decay rate
    ix::Int
    iu::Int

    iq::Int
    iy::Int
    ip::Int

    Cn::Matrix # commutation matrix
    Cm::Matrix # commutation matrix
    function LinearDLMI(alpha,ix,iu)
        Cn = com_mat(ix,ix)
        Cm = com_mat(iu,ix)
        # ip = ix*ix# + 1 # Z11, b
        ip = 0
        new(alpha,ix,iu,ix*ix,ix*iu,ip,Cn,Cm)
    end
end

function forward(model::LinearDLMI, q::Vector, y::Vector, z::Vector, A::Matrix, B::Matrix)
    Q = reshape(q,(model.ix,model.ix))
    Y = reshape(y,(model.iu,model.ix))
    Z = reshape(z,(model.ix,model.ix))
   
    L = A*Q + B*Y
    dQ = L + L' + model.alpha*Q + Z
    return vec(dQ)
end

function diff(model::LinearDLMI,A::Matrix,B::Matrix)
    Imat = I(model.ix)
    Aq = kron(Imat,A) + kron(A,Imat) * model.Cn + model.alpha * kron(Imat,Imat)
    Bq = kron(Imat,B) + kron(B,Imat) * model.Cm
    Sq = kron(Imat,Imat)
    return Aq,Bq,Sq
end

function discretize_foh(model::FunnelDynamics,dynamics::Dynamics,
        x::Matrix,u::Matrix,T::Vector,
        Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3})
    @assert size(x,2) == size(Q,3)
    @assert size(x,2) + 1 == size(u,2)
    @assert size(u,2) == size(Y,3)
    @assert size(Y,3) == size(Z,3)

    N = size(x,2)
    ix = model.ix
    iq = model.iq
    iy = model.iy

    idx_x = 1:ix
    idx_q = (ix+1):(ix+iq)
    idx_A = (ix+iq+1):(ix+iq+iq*iq)
    idx_Bm = (ix+iq+iq*iq+1):(ix+iq+iq*iq+iq*iy)
    idx_Bp = (ix+iq+iq*iq+iq*iy+1):(ix+iq+iq*iq+iq*iy+iq*iy)
    idx_Sm = (ix+iq+iq*iq+iq*iy+iq*iy+1):(ix+iq+iq*iq+iq*iy+iq*iy+iq*iq)
    idx_Sp = (ix+iq+iq*iq+iq*iy+iq*iy+iq*iq+1):(ix+iq+iq*iq+iq*iy+iq*iy+iq*iq+iq*iq)

    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        ym = p[3]
        yp = p[4]
        zm = p[5]
        zp = p[6]
        dt = p[7]

        alpha = (dt - t) / dt
        beta = t / dt

        u_ = alpha * um + beta * up
        y_ = alpha * ym + beta * yp
        z_ = alpha * zm + beta * zp

        x_ = V[idx_x]
        q_ = V[idx_q]
        Phi = reshape(V[idx_A], (iq, iq))
        Bm_ = reshape(V[idx_Bm],(iq,iy))
        Bp_ = reshape(V[idx_Bp],(iq,iy))
        Sm_ = reshape(V[idx_Sm],(iq,iq))
        Sp_ = reshape(V[idx_Sp],(iq,iq))

        # traj terms
        f = forward(dynamics,x_,u_)
        A,B = diff(dynamics,x_,u_)
        # funl terms
        F = forward(model,q_,y_,z_,A,B)
        Aq_,Bq,Sq = diff(model,A,B)

        dxdt = f
        dqdt = F
        dAdt = Aq_*Phi
        dBmdt = Aq_*Bm_ + Bq.*alpha
        dBpdt = Aq_*Bp_ + Bq.*beta
        if typeof(model) == LinearFOH
            dSmdt = Aq_*Sm_ + Sq
            dSpdt = dSmdt .* 0
        else
            dSmdt = Aq_*Sm_ + Sq.*alpha
            dSpdt = Aq_*Sp_ + Sq.*beta
        end
        dV = [dxdt;dqdt;dAdt[:];dBmdt[:];dBpdt[:];dSmdt[:];dSpdt[:]]
        out .= dV[:]
    end

    Aq = zeros(iq,iq,N)
    Bm = zeros(iq,iy,N)
    Bp = zeros(iq,iy,N)
    Sm = zeros(iq,iq,N)
    Sp = zeros(iq,iq,N)

    x_prop = zeros(ix,N)
    q_prop = zeros(iq,N)

    for i = 1:N
        A0 = I(iq)
        Bm0 = zeros(iq,iy)
        Bp0 = zeros(iq,iy)
        Sm0 = zeros(iq,iq)
        Sp0 = zeros(iq,iq)
        V0 = [x[:,i];vec(Q[:,:,i]);A0[:];Bm0[:];Bp0[:];Sm0[:];Sp0[:]][:]

        um = u[:,i]
        up = u[:,i+1]
        ym = vec(Y[:,:,i])
        yp = vec(Y[:,:,i+1])
        zm = vec(Z[1:ix,:,i])
        if typeof(model) == LinearFOH
            zp = vec(Z[:,:,i])
        elseif typeof(model) == LinearSOH
            zp = vec(Z[ix+1:2*ix,:,i])
        else
            zp = vec(Z[:,:,i+1])
        end
        dt = T[i]

        t, sol = RK4(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,dt),50)
        x_prop[:,i] .= sol[idx_x,end]
        q_prop[:,i] .= sol[idx_q,end]
        Aq[:,:,i] .= reshape(sol[idx_A,end],iq,iq)
        Bm[:,:,i] .= reshape(sol[idx_Bm,end],iq,iy)
        Bp[:,:,i] .= reshape(sol[idx_Bp,end],iq,iy)
        Sm[:,:,i] .= reshape(sol[idx_Sm,end],iq,iq)
        Sp[:,:,i] .= reshape(sol[idx_Sp,end],iq,iq)
    end
    return Aq,Bm,Bp,Sm,Sp,x_prop,q_prop
end


# struct LinearQZ <: FunnelDynamics
#     alpha::Float64 # decay rate
#     ix::Int
#     iu::Int

#     iq::Int
#     iy::Int

#     Cn::Matrix # commutation matrix
#     Cm::Matrix # commutation matrix
#     function LinearQZ(alpha,ix,iu)
#         Cn = com_mat(ix,ix)
#         Cm = com_mat(iu,ix)
#         new(alpha,ix,iu,ix*ix,ix*iu,Cn,Cm)
#     end
# end

# function forward(model::LinearQZ, q::Vector, k::Vector, z::Vector, A::Matrix, B::Matrix)
#     Q = reshape(q,(model.ix,model.ix))
#     K = reshape(k,(model.iu,model.ix))
#     Z = reshape(z,(model.ix,model.ix))
   
#     L = (A + B*K)*Q
#     dQ = L + L' + model.alpha*Q + Z
#     return vec(dQ)
# end

# function diff(model::LinearQZ,k::Vector,A::Matrix,B::Matrix)
#     K = reshape(k,(model.iu,model.ix))
#     Imat = I(model.ix)
#     Aq = kron(Imat,A+B*K) + kron(A+B*K,Imat) * model.Cn + model.alpha * kron(Imat,Imat)
#     # Bq = kron(Imat,B) + kron(B,Imat) * model.Cm
#     Bz = kron(Imat,Imat)
#     return Aq,Bz
# end

# function discretize_foh(model::LinearQZ,dynamics::Dynamics,
#         x::Matrix,u::Matrix,T::Vector,
#         Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3})
#     # Q,Y,Z are unnecessary for linear DLMI, but keep them here for writing nonlinearDLMI later
#     @assert size(x,2) == size(Q,3)
#     @assert size(x,2) + 1 == size(u,2)
#     @assert size(u,2) == size(Y,3)
#     @assert size(Y,3) == size(Z,3)

#     N = size(x,2)
#     ix = model.ix
#     iq = model.iq
#     iy = model.iy

#     idx_x = 1:ix
#     idx_q = (ix+1):(ix+iq)
#     idx_A = (ix+iq+1):(ix+iq+iq*iq)
#     idx_Bm = (ix+iq+iq*iq+1):(ix+iq+iq*iq+iq*iq)
#     idx_Bp = (ix+iq+iq*iq+iq*iq+1):(ix+iq+iq*iq+iq*iq+iq*iq)

#     function dvdt(out,V,p,t)
#         um = p[1]
#         up = p[2]
#         ym = p[3]
#         yp = p[4]
#         zm = p[5]
#         zp = p[6]
#         dt = p[7]

#         alpha = (dt - t) / dt
#         beta = t / dt

#         u_ = alpha * um + beta * up
#         y_ = alpha * ym + beta * yp
#         z_ = alpha * zm + beta * zp

#         x_ = V[idx_x]
#         q_ = V[idx_q]
#         Phi = reshape(V[idx_A], (iq, iq))
#         Bm_ = reshape(V[idx_Bm],(iq,iq))
#         Bp_ = reshape(V[idx_Bp],(iq,iq))

#         # traj terms
#         f = forward(dynamics,x_,u_)
#         A,B = diff(dynamics,x_,u_)
#         # funl terms
#         F = forward(model,q_,y_,z_,A,B)
#         Aq_,Bq, = diff(model,y_,A,B)

#         dxdt = f
#         dqdt = F
#         dAdt = Aq_*Phi
#         dBmdt = Aq_*Bm_ + Bq.*alpha
#         dBpdt = Aq_*Bp_ + Bq.*beta
#         dV = [dxdt;dqdt;dAdt[:];dBmdt[:];dBpdt[:]]
#         out .= dV[:]
#     end

#     Aq = zeros(iq,iq,N)
#     Bm = zeros(iq,iq,N)
#     Bp = zeros(iq,iq,N)

#     x_prop = zeros(ix,N)
#     q_prop = zeros(iq,N)

#     for i = 1:N
#         A0 = I(iq)
#         Bm0 = zeros(iq,iq)
#         Bp0 = zeros(iq,iq)
#         V0 = [x[:,i];vec(Q[:,:,i]);A0[:];Bm0[:];Bp0[:]][:]

#         um = u[:,i]
#         up = u[:,i+1]
#         ym = vec(Y[:,:,i])
#         yp = vec(Y[:,:,i+1])
#         zm = vec(Z[:,:,i])
#         zp = vec(Z[:,:,i+1])
#         dt = T[i]

#         t, sol = RK4(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,dt),50)
#         x_prop[:,i] .= sol[idx_x,end]
#         q_prop[:,i] .= sol[idx_q,end]
#         Aq[:,:,i] .= reshape(sol[idx_A,end],iq,iq)
#         Bm[:,:,i] .= reshape(sol[idx_Bm,end],iq,iq)
#         Bp[:,:,i] .= reshape(sol[idx_Bp,end],iq,iq)
#     end
#     return Aq,Bm,Bp,x_prop,q_prop
# end

# struct LinearQS <: FunnelDynamics
#     alpha::Float64 # decay rate
#     ix::Int
#     iu::Int

#     iq::Int
#     iy::Int

#     Cn::Matrix # commutation matrix
#     function LinearQS(alpha,ix,iu)
#         Cn = com_mat(ix,ix)
#         new(alpha,ix,iu,ix*ix,iu*iu,Cn)
#     end
# end


# function forward(model::LinearQS, q::Vector, y::Vector, z::Vector, A::Matrix, B::Matrix)
#     Q = reshape(q,(model.ix,model.ix))
#     Y = reshape(y,(model.iu,model.iu))
#     Z = reshape(z,(model.ix,model.ix))
   
#     L = A*Q
#     # dQ = L + L' - B*Y*B' + model.alpha*Q + Z
#     dQ = A*Q + Q*A' - B*Y*B' + model.alpha*Q + Z
#     return vec(dQ)
# end

# function diff(model::LinearQS,A::Matrix,B::Matrix)
#     Imat = I(model.ix)
#     # Aq = kron(Imat,A) + kron(A,Imat) * model.Cn + model.alpha * kron(Imat,Imat)
#     Aq = kron(Imat,A) + kron(A,Imat) + model.alpha * kron(Imat,Imat)
#     Bq = - kron(B,B)
#     Sq = kron(Imat,Imat)
#     return Aq,Bq,Sq
# end