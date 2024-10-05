include("../trajopt/dynamics.jl")
include("../funlopt/funl_dynamics.jl")
include("../trajopt/discretize.jl")
using LinearAlgebra

abstract type FunnelCTCS end

struct ConcatCTCS <: FunnelCTCS
    is::Int
    list_CTCS::Vector{FunnelCTCS}
    function ConcatCTCS(list_CTCS::Vector{FunnelCTCS})
        is = length(list_CTCS)
        new(is,list_CTCS)
    end
end

function forward(model::ConcatCTCS, q::Vector, y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    ans = zeros(model.is)
    for i in 1:model.is
        ans[i] = forward(model.list_CTCS[i],q,y,z,b,A,B)
    end
    return ans
end

function diff(model::ConcatCTCS,q::Vector, y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    iq = size(q,1)
    iy = size(y,1)
    iz = size(z,1)

    Aq = zeros(model.is,iq)
    Sy = zeros(model.is,iy)
    Sz = zeros(model.is,iq)
    Sb = zeros(model.is,1)

    for i in 1:model.is
        Aq[i,:],Sy[i,:],Sz[i,:],Sb[i,:] = diff(model.list_CTCS[i],q,y,z,b,A,B)
    end
    return Aq,Sy,Sz,Sb
end

struct Invariance <: FunnelCTCS
    M::Float64
    epsilon::Float64
    is::Int
    θ::Float64
    dynamics::Any
    funl_dynamics::Any

    Cq::Matrix
    Cy::Matrix
    alpha::Float64
    function Invariance(M,epsilon,alpha,θ,dynamics,funl_dynamics)
        @assert M >= 0
        @assert epsilon >= 0
        # @assert typeof(funl_dynamics) == LinearFOH || typeof(funl_dynamics) == LinearSOH
        is = 1
        Cq = com_mat(ix,ix)
        Cy = com_mat(iu,ix)
        @assert alpha > 1.0
        new(M,epsilon,is,θ,dynamics,funl_dynamics,Cq,Cy,alpha)
    end
end

function forward(model::Invariance, q::Vector, y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    ix = model.dynamics.ix
    iu = model.dynamics.iu
    iμ = model.dynamics.iμ
    iψ = model.dynamics.iψ
    θ = model.θ

    Q = reshape(q,(ix,ix))
    Y = reshape(y,(iu,ix))
    Z = reshape(z,(ix,ix))

    N11 = diagm(θ ./ ( model.dynamics.β .* model.dynamics.β))
    N22 =  b .* θ .* Matrix{Float64}(I, iψ, iψ)

    if typeof(model.funl_dynamics) == LinearDLMI
        LMI11 = - Z
    else
        LMI11 = A*Q + Q'*A' + B*Y + Y'*B' + model.funl_dynamics.alpha.*0.5.*(Q+Q') - Z
    end
    LMI21 = N22 * model.dynamics.G'
    LMI22 = -N22
    LMI31 = model.dynamics.Cμ * Q + model.dynamics.Dμu * Y
    LMI32 = zeros(iμ,iψ)
    LMI33 = -N11
    LMI = [LMI11 LMI21' LMI31';
        LMI21 LMI22 LMI32';
        LMI31 LMI32 LMI33
    ]
    iLMI = size(LMI,1)
    epsilon = model.epsilon * Matrix{Float64}(I, iLMI, iLMI)
    H = LMI + epsilon
    # eigval,eigvec = eigen(Symmetric(H))
    eigval,eigvec = eigen(H)
    g = eigval[end]
    # return max(0,model.M*g)^2
    return max(0,model.M*g)^model.alpha
end

function diff(model::Invariance,q::Vector, y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    ix = model.dynamics.ix
    iu = model.dynamics.iu
    iμ = model.dynamics.iμ
    iψ = model.dynamics.iψ
    θ = model.θ

    iq = size(q,1)
    iy = size(y,1)
    iz = size(z,1)

    Aq = zeros(model.is,iq)
    Sy = zeros(model.is,iy)
    Sz = zeros(model.is,iq)
    Sb = zeros(model.is,1)

    Q = reshape(q,(ix,ix))
    Y = reshape(y,(iu,ix))
    Z = reshape(z,(ix,ix))

    N11 = diagm(θ ./ ( model.dynamics.β .* model.dynamics.β))
    N22 =  b .* θ .* Matrix{Float64}(I, iψ, iψ)
    if typeof(model.funl_dynamics) == LinearDLMI
        LMI11 = - Z
    else
        LMI11 = A*Q + Q'*A' + B*Y + Y'*B' + model.funl_dynamics.alpha.*0.5.*(Q+Q') - Z
    end
    LMI21 = N22 * model.dynamics.G'
    LMI22 = -N22
    LMI31 = model.dynamics.Cμ * Q + model.dynamics.Dμu * Y
    LMI32 = zeros(iμ,iψ)
    LMI33 = -N11
    LMI = [LMI11 LMI21' LMI31';
        LMI21 LMI22 LMI32';
        LMI31 LMI32 LMI33
    ]
    iLMI = size(LMI,1)
    epsilon = model.epsilon * Matrix{Float64}(I, iLMI, iLMI)
    H = LMI + epsilon
    eigval,eigvec = eigen(H)
    g = eigval[end]
    if g >= 0
        # dfg = 2*(model.M^2)*g
        dfg = (model.alpha)*(model.M^model.alpha)*g^(model.alpha-1)
    else
        dfg = 0
    end
    dλdH = vec(eigvec[:,end] * eigvec[:,end]')'

    if typeof(model.funl_dynamics) == LinearDLMI
        FQ = [zeros(ix,ix);zeros(iψ,ix);model.dynamics.Cμ]
    else
        FQ = [A .+ 0.5*model.funl_dynamics.alpha .* Matrix(1.0I,ix,ix);zeros(iψ,ix);model.dynamics.Cμ]
    end
    RQ = [Matrix(1.0I,ix,ix) zeros(ix,iψ) zeros(ix,iμ)]
    dHdq = kron(RQ',FQ) + kron(FQ,RQ') * model.Cq
    Aq .= dfg .* dλdH * dHdq

    if typeof(model.funl_dynamics) == LinearDLMI
        FY = [zeros(ix,iu);zeros(iψ,iu);model.dynamics.Dμu]
    else
        FY = [B;zeros(iψ,iu);model.dynamics.Dμu]
    end
    RY = [I(ix) zeros(ix,iψ) zeros(ix,iμ)]
    dHdy = kron(RY',FY) + kron(FY,RY') * model.Cy
    Sy .= dfg .* dλdH * dHdy

    FZ = [I(ix);zeros(iψ,ix);zeros(iμ,ix)]
    dHdz = - kron(FZ,FZ)
    Sz .= dfg .* dλdH * dHdz

    N11 = zeros(iμ,iμ)
    N22 =  θ .* Matrix{Float64}(I, iψ, iψ)
    LMI11 = zeros(ix,ix)
    LMI21 = N22 * model.dynamics.G'
    LMI22 = -N22
    LMI31 = zeros(iμ,ix)
    LMI32 = zeros(iμ,iψ)
    LMI33 = -N11
    LMI = [LMI11 LMI21' LMI31';
        LMI21 LMI22 LMI32';
        LMI31 LMI32 LMI33
    ]
    dHdb = vec(LMI)
    Sb .= dfg .* dλdH * dHdb

    return Aq,Sy,Sz,Sb
end

struct BoundB <: FunnelCTCS
    M::Float64
    epsilon::Float64
    is::Int
    θ::Float64
    dynamics::Any
    funl_dynamics::Any

    Cq::Matrix
    Cy::Matrix
    
    alpha::Float64
    function BoundB(M,epsilon,alpha,θ,dynamics,funl_dynamics)
        @assert M >= 0
        @assert epsilon >= 0
        # @assert typeof(funl_dynamics) == LinearFOH || typeof(funl_dynamics) == LinearSOH
        is = 1
        Cq = com_mat(ix,ix)
        Cy = com_mat(iu,ix)
        @assert alpha > 1.0
        new(M,epsilon,is,θ,dynamics,funl_dynamics,Cq,Cy,alpha)
    end
end

function forward(model::BoundB,q::Vector,y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    ix = model.dynamics.ix
    iu = model.dynamics.iu
    iv = model.dynamics.iv
    Cv = model.dynamics.Cv
    Dvu = model.dynamics.Dvu
    # θ = model.θ

    Q = reshape(q,(ix,ix))
    Y = reshape(y,(iu,ix))
    # Z = reshape(z,(ix,ix))

    tmp12 = Cv*Q + Dvu*Y
    LMI =  - [b * I(iv) tmp12;
        tmp12' Q
    ]

    iLMI = size(LMI,1)
    epsilon = model.epsilon * Matrix{Float64}(I, iLMI, iLMI)
    H = LMI + epsilon
    eigval,eigvec = eigen(H)
    g = eigval[end]
    return max(0,model.M*g)^model.alpha
    # return max(0,model.M*g)^2
end

function diff(model::BoundB,q::Vector, y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    ix = model.dynamics.ix
    iu = model.dynamics.iu
    iv = model.dynamics.iv
    Cv = model.dynamics.Cv
    Dvu = model.dynamics.Dvu

    iq = size(q,1)
    iy = size(y,1)
    iz = size(z,1)

    Aq = zeros(model.is,iq)
    Sy = zeros(model.is,iy)
    Sz = zeros(model.is,iq)
    Sb = zeros(model.is,1)

    Q = reshape(q,(ix,ix))
    Y = reshape(y,(iu,ix))
    Z = reshape(z,(ix,ix))

    tmp12 = Cv*Q + Dvu*Y
    LMI =  - [b * I(iv) tmp12;
        tmp12' Q
    ]
    iLMI = size(LMI,1)
    epsilon = model.epsilon * Matrix{Float64}(I, iLMI, iLMI)
    H = LMI + epsilon
    eigval,eigvec = eigen(H)
    g = eigval[end]
    if g >= 0
        dfg = (model.alpha)*(model.M^model.alpha)*g^(model.alpha-1)
        # dfg = (model.alpha)*(model.M^(model.alpha-1))*g^(model.alpha-1)
    else
        dfg = 0
    end
    dλdH = vec(eigvec[:,end] * eigvec[:,end]')'

    FQ = [model.dynamics.Cv;0.5*Matrix(1.0I,ix,ix)]
    RQ = [zeros(ix,iv) Matrix(1.0I,ix,ix)]
    dHdq = - (kron(RQ',FQ) + kron(FQ,RQ') * model.Cq)
    Aq .= dfg .* dλdH * dHdq

    FY = [model.dynamics.Dvu;zeros(ix,iu)]
    RY = [zeros(ix,iv) Matrix(1.0I,ix,ix)]
    dHdy = - (kron(RY',FY) + kron(FY,RY') * model.Cy)
    Sy .= dfg .* dλdH * dHdy

    # tmp12 = fs.dynamics.Cv*Q + fs.dynamics.Dvu*Y
    LMI =  - [1.0 * I(iv) zeros(iv,ix);
        zeros(ix,iv) zeros(ix,ix)
    ]
    dHdb = vec(LMI)
    Sb .= dfg .* dλdH * dHdb

    return Aq,Sy,Sz,Sb
end

function diff_numeric(model::FunnelCTCS,q::Vector, y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    iq = size(q,1)
    iy = size(y,1)
    iz = size(z,1)

    Aq = zeros(model.is,iq)
    Sy = zeros(model.is,iy)
    Sz = zeros(model.is,iq)
    Sb = zeros(model.is,1)

    h = 1e-6
    eps_q = Matrix{Float64}(I,iq,iq)
    for i in 1:iq
        Aq[:,i] .= (
        forward(model,q+h*eps_q[:,i],y,z,b,A,B) -
        forward(model,q-h*eps_q[:,i],y,z,b,A,B)
        ) / (2*h)
    end

    eps_y = Matrix{Float64}(I,iy,iy)
    for i in 1:iy
        Sy[:,i] .= (
        forward(model,q,y+h*eps_y[:,i],z,b,A,B) -
        forward(model,q,y-h*eps_y[:,i],z,b,A,B)
        ) / (2*h)
    end

    eps_z = Matrix{Float64}(I,iz,iz)
    for i in 1:iz
        Sz[:,i] .= (
        forward(model,q,y,z+h*eps_z[:,i],b,A,B) -
        forward(model,q,y,z-h*eps_z[:,i],b,A,B)
        ) / (2*h)
    end

    Sb[:,1] .= (
        forward(model,q,y,z,b+h,A,B) -
        forward(model,q,y,z,b-h,A,B)
        ) / (2*h)
    return Aq,Sy,Sz,Sb
end

function discretize_foh(model::FunnelDynamics,dynamics::Dynamics,ctcs::FunnelCTCS,
    x::Matrix,u::Matrix,T::Vector,
    Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3},b::Vector)
    # Q,Y,Z are unnecessary for linear DLMI, but keep them here for writing nonlinearDLMI later
    @assert size(x,2) == size(Q,3)
    @assert size(x,2) + 1 == size(u,2)
    @assert size(u,2) == size(Y,3)
    @assert size(Y,3) == size(Z,3)

    N = size(x,2)
    ix = model.ix
    iq = model.iq
    iy = model.iy
    is = ctcs.is
    function get_interval(start,size)
        return start:(start+size-1)
    end

    idx_x = get_interval(1,ix)
    idx_q = get_interval(idx_x[end]+1,iq)
    idx_A = get_interval(idx_q[end]+1,iq*iq)
    idx_Bym = get_interval(idx_A[end]+1,iq*iy)
    idx_Byp = get_interval(idx_Bym[end]+1,iq*iy)
    idx_Bzm = get_interval(idx_Byp[end]+1,iq*iq)
    idx_Bzp = get_interval(idx_Bzm[end]+1,iq*iq)
    
    idx_s = get_interval(idx_Bzp[end]+1,is)
    idx_Sq = get_interval(idx_s[end]+1,is*iq)
    idx_Sym = get_interval(idx_Sq[end]+1,is*iy)
    idx_Syp = get_interval(idx_Sym[end]+1,is*iy)
    idx_Szm = get_interval(idx_Syp[end]+1,is*iq)
    idx_Szp = get_interval(idx_Szm[end]+1,is*iq)
    idx_Sbm = get_interval(idx_Szp[end]+1,is*1)
    idx_Sbp = get_interval(idx_Sbm[end]+1,is*1)

    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        ym = p[3]
        yp = p[4]
        zm = p[5]
        zp = p[6]
        bm = p[7]
        bp = p[8]
        dt = p[9]

        alpha = (dt - t) / dt
        beta = t / dt

        u_ = alpha * um + beta * up
        y_ = alpha * ym + beta * yp
        z_ = alpha * zm + beta * zp
        b_ = alpha * bm + beta * bp

        x_ = V[idx_x]
        q_ = V[idx_q]
        Phi = reshape(V[idx_A], (iq, iq))
        Bym_ = reshape(V[idx_Bym],(iq,iy))
        Byp_ = reshape(V[idx_Byp],(iq,iy))
        Bzm_ = reshape(V[idx_Bzm],(iq,iq))
        Bzp_ = reshape(V[idx_Bzp],(iq,iq))

        s_ = V[idx_s]
        Sq_ = reshape(V[idx_Sq], (is, iq))
        Sym_ = reshape(V[idx_Sym],(is,iy))
        Syp_ = reshape(V[idx_Syp],(is,iy))
        Szm_ = reshape(V[idx_Szm],(is,iq))
        Szp_ = reshape(V[idx_Szp],(is,iq))
        Sbm_ = reshape(V[idx_Sbm],(is,1))
        Sbp_ = reshape(V[idx_Sbp],(is,1))

        # traj terms
        f = forward(dynamics,x_,u_)
        A,B = diff(dynamics,x_,u_)
        # funl terms
        F = forward(model,q_,y_,z_,A,B)
        dAq,dBy,dBz = diff(model,A,B)
        # ctcs terms
        ds = forward(ctcs,q_,y_,z_,b_,A,B)
        dSq,dSy,dSz,dSb = diff(ctcs,q_,y_,z_,b_,A,B)

        dxdt = f
        dqdt = F
        dAdt = dAq*Phi
        dBymdt = dAq*Bym_ + dBy.*alpha
        dBypdt = dAq*Byp_ + dBy.*beta
        if typeof(model) == LinearFOH
            dBzmdt = dAq*Bzm_ + dBz
            dBzpdt = dBzmdt .* 0
        else
            dBzmdt = dAq*Bzm_ + dBz.*alpha
            dBzpdt = dAq*Bzp_ + dBz.*beta
        end
        
        dsdt = ds
        dSqdt = dSq*Phi
        dSymdt = dSq*Bym_ + dSy.*alpha
        dSypdt = dSq*Byp_ + dSy.*beta
        if typeof(model) == LinearFOH
            dSzmdt = dSq*Bzm_ + dSz
            dSzpdt = dSzmdt .* 0
        else
            dSzmdt = dSq*Bzm_ + dSz .* alpha
            dSzpdt = dSq*Bzp_ + dSz .* beta
        end
        dSbmdt = dSb.*alpha
        dSbpdt = dSb.*beta

        dV = [dxdt;dqdt;
            dAdt[:];dBymdt[:];dBypdt[:];
            dBzmdt[:];dBzpdt[:];
            dsdt;dSqdt[:];
            dSymdt[:];dSypdt[:];
            dSzmdt[:];dSzpdt[:];
            dSbmdt[:];dSbpdt[:]]
        out .= dV[:]
    end

    Aq = zeros(iq,iq,N)
    Bym = zeros(iq,iy,N)
    Byp = zeros(iq,iy,N)
    Bzm = zeros(iq,iq,N)
    Bzp = zeros(iq,iq,N)

    Sq = zeros(is,iq,N)
    Sym = zeros(is,iy,N)
    Syp = zeros(is,iy,N)
    Szm = zeros(is,iq,N)
    Szp = zeros(is,iq,N)
    Sbm = zeros(is,1,N)
    Sbp = zeros(is,1,N)
    SZ = zeros(is,N)

    x_prop = zeros(ix,N)
    q_prop = zeros(iq,N)
    s_prop = zeros(is,N)

    for i = 1:N
        A0 = I(iq)
        Bym0 = zeros(iq,iy)
        Byp0 = zeros(iq,iy)
        Bzm0 = zeros(iq,iq)
        Bzp0 = zeros(iq,iq)
        S0 = zeros(is,iq)
        Sym0 = zeros(is,iy)
        Syp0 = zeros(is,iy)
        Szm0 = zeros(is,iq)
        Szp0 = zeros(is,iq)
        Sbm0 = zeros(is,1)
        Sbp0 = zeros(is,1)
        V0 = [x[:,i];vec(Q[:,:,i]);A0[:];Bym0[:];Byp0[:];
            Bzm0[:];Bzp0[:];
            zeros(is);S0[:];Sym0[:];Syp0[:];
            Szm0[:];Szp0[:];
            Sbm0[:];Sbp0[:]][:]

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
        bm = b[i]
        bp = b[i+1]
        dt = T[i]

        t, sol = RK4(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,bm,bp,dt),50)
        x_prop[:,i] .= sol[idx_x,end]
        q_prop[:,i] .= sol[idx_q,end]
        Aq[:,:,i] .= reshape(sol[idx_A,end],iq,iq)
        Bym[:,:,i] .= reshape(sol[idx_Bym,end],iq,iy)
        Byp[:,:,i] .= reshape(sol[idx_Byp,end],iq,iy)
        Bzm[:,:,i] .= reshape(sol[idx_Bzm,end],iq,iq)
        Bzp[:,:,i] .= reshape(sol[idx_Bzp,end],iq,iq)
        s_prop[:,i] .= sol[idx_s,end]
        Sq[:,:,i] .= reshape(sol[idx_Sq,end],is,iq)
        Sym[:,:,i] .= reshape(sol[idx_Sym,end],is,iy)
        Syp[:,:,i] .= reshape(sol[idx_Syp,end],is,iy)
        Szm[:,:,i] .= reshape(sol[idx_Szm,end],is,iq)
        Szp[:,:,i] .= reshape(sol[idx_Szp,end],is,iq)
        Sbm[:,:,i] .= reshape(sol[idx_Sbm,end],is,1)
        Sbp[:,:,i] .= reshape(sol[idx_Sbp,end],is,1)
        SZ[:,i] .= (s_prop[:,i] - Sq[:,:,i]*vec(Q[:,:,i]) 
            - Sym[:,:,i]*ym - Syp[:,:,i]*yp
            - Szm[:,:,i]*zm - Szp[:,:,i]*zp
            - Sbm[:,:,i]*bm - Sbp[:,:,i]*bp)
    end
    return Aq,Bym,Byp,Bzm,Bzp,Sq,Sym,Syp,Szm,Szp,Sbm,Sbp,SZ,x_prop,q_prop,s_prop
end

struct QPD <: FunnelCTCS
    M::Float64
    epsilon::Float64
    is::Int
    dynamics::Any
    funl_dynamics::Any

    alpha::Float64
    function QPD(M,epsilon,alpha,dynamics,funl_dynamics)
        @assert M >= 0
        @assert epsilon >= 0
        @assert typeof(funl_dynamics) == LinearDLMI
        is = 1
        @assert alpha > 1.0
        new(M,epsilon,is,dynamics,funl_dynamics,alpha)
    end
end

function forward(model::QPD, q::Vector, y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    ix = model.dynamics.ix

    Q = reshape(q,(ix,ix))
    LMI = -Q
    iLMI = size(LMI,1)
    epsilon = model.epsilon * Matrix{Float64}(I, iLMI, iLMI)
    H = LMI + epsilon
    eigval,eigvec = eigen(H)
    g = eigval[end]
    return max(0,model.M*g)^model.alpha
end

# function diff(model::QPD, q::Vector, y::Vector, z::Vector, b::Float64)
function diff(model::QPD,q::Vector,y::Vector,z::Vector,b::Float64,A::Matrix,B::Matrix)
    ix = model.dynamics.ix

    iq = size(q,1)
    iy = size(y,1)
    iz = size(z,1)

    Aq = zeros(model.is,iq)
    Sy = zeros(model.is,iy)
    Sz = zeros(model.is,iq)
    Sb = zeros(model.is,1)
    
    Q = reshape(q,(ix,ix))
    LMI = -Q
    iLMI = size(LMI,1)
    epsilon = model.epsilon * Matrix{Float64}(I, iLMI, iLMI)
    H = LMI + epsilon
    eigval,eigvec = eigen(H)
    g = eigval[end]
    if g >= 0
        dfg = (model.alpha)*(model.M^model.alpha)*g^(model.alpha-1)
    else
        dfg = 0
    end
    dλdH = vec(eigvec[:,end] * eigvec[:,end]')'

    dHdq = - kron(Matrix(1.0I,ix,ix),Matrix(1.0I,ix,ix))
    Aq .= dfg .* dλdH * dHdq
    return Aq,Sy,Sz,Sb
end

struct Invariance <: FunnelCTCS
    M::Float64
    epsilon::Float64
    is::Int
    θ::Float64
    dynamics::Any
    funl_dynamics::Any

    Cq::Matrix
    Cy::Matrix
    alpha::Float64
    function Invariance(M,epsilon,alpha,θ,dynamics,funl_dynamics::LinearDLMI)
        @assert M >= 0
        @assert epsilon >= 0
        # @assert typeof(funl_dynamics) == LinearFOH || typeof(funl_dynamics) == LinearSOH
        is = 1
        Cq = com_mat(ix,ix)
        Cy = com_mat(iu,ix)
        @assert alpha > 1.0
        new(M,epsilon,is,θ,dynamics,funl_dynamics,Cq,Cy,alpha)
    end
end

# function forward(model::HLsmooth, q::Vector, y::Vector,z::Vector,b::Float64)
#     # -M - logdetQ
#     ix = model.dynamics.ix
#     iu = model.dynamics.iu
#     iμ = model.dynamics.iμ
#     iψ = model.dynamics.iψ
#     θ = model.θ

#     Q = reshape(q,(ix,ix))
#     Y = reshape(y,(iu,ix))
#     Z = reshape(z,(ix,ix))

#     N11 = diagm(θ ./ ( model.dynamics.β .* model.dynamics.β))
#     N22 =  b .* θ .* Matrix{Float64}(I, iψ, iψ)

#     LMI11 = -Z
#     LMI21 = N22 * model.dynamics.G'
#     LMI22 = -N22
#     LMI31 = model.dynamics.Cμ * Q + model.dynamics.Dμu * Y
#     LMI32 = zeros(iμ,iψ)
#     LMI33 = -N11
#     LMI = [LMI11 LMI21' LMI31';
#         LMI21 LMI22 LMI32';
#         LMI31 LMI32 LMI33
#     ]
#     iLMI = size(LMI,1)
#     epsilon = model.epsilon * Matrix{Float64}(I, iLMI, iLMI)
#     H = LMI + epsilon
#     # eigval,eigvec = eigen(Symmetric(H))
#     eigval,eigvec = eigen(H)
#     g = eigval[end]
#     return max(0,model.M*g)^2
# end

# function diff(model::HLsmooth,q::Vector, y::Vector,z::Vector,b::Float64)
#     ix = model.dynamics.ix
#     iu = model.dynamics.iu
#     iμ = model.dynamics.iμ
#     iψ = model.dynamics.iψ
#     θ = model.θ

#     iq = size(q,1)
#     iy = size(y,1)
#     iz = size(z,1)

#     Aq = zeros(model.is,iq)
#     Sy = zeros(model.is,iy)
#     Sz = zeros(model.is,iq)
#     Sb = zeros(model.is,1)

#     Q = reshape(q,(ix,ix))
#     Y = reshape(y,(iu,ix))
#     Z = reshape(z,(ix,ix))

#     N11 = diagm(θ ./ ( model.dynamics.β .* model.dynamics.β))
#     N22 =  b .* θ .* Matrix{Float64}(I, iψ, iψ)
#     LMI11 = -Z
#     LMI21 = N22 * model.dynamics.G'
#     LMI22 = -N22
#     LMI31 = model.dynamics.Cμ * Q + model.dynamics.Dμu * Y
#     LMI32 = zeros(iμ,iψ)
#     LMI33 = -N11
#     LMI = [LMI11 LMI21' LMI31';
#         LMI21 LMI22 LMI32';
#         LMI31 LMI32 LMI33
#     ]
#     iLMI = size(LMI,1)
#     epsilon = model.epsilon * Matrix{Float64}(I, iLMI, iLMI)
#     H = LMI + epsilon
#     # eigval,eigvec = eigen(Symmetric(H))
#     eigval,eigvec = eigen(H)
#     g = eigval[end]
#     if g >= 0
#         # dfg = 1
#         dfg = 2*g
#     else
#         dfg = 0
#     end
#     dλdH = model.M .* vec(eigvec[:,end] * eigvec[:,end]')'

#     FQ = [zeros(ix,ix);zeros(iψ,ix);model.dynamics.Cμ]
#     RQ = [I(ix) zeros(ix,iψ) zeros(ix,iμ)]
#     dHdq = kron(RQ',FQ) + kron(FQ,RQ') * model.Cq
#     Aq .= dfg .* dλdH * dHdq

#     FY = [zeros(ix,iu);zeros(iψ,iu);model.dynamics.Dμu]
#     RY = [I(ix) zeros(ix,iψ) zeros(ix,iμ)]
#     dHdy = kron(RY',FY) + kron(FY,RY') * model.Cy
#     Sy .= dfg .* dλdH * dHdy

#     FZ = [I(ix);zeros(iψ,ix);zeros(iμ,ix)]
#     dHdz = kron(FZ,FZ)
#     Sz .= dfg .* dλdH * (-dHdz)

#     N11 = zeros(iμ,iμ)
#     N22 =  θ .* Matrix{Float64}(I, iψ, iψ)
#     LMI11 = zeros(ix,ix)
#     LMI21 = N22 * model.dynamics.G'
#     LMI22 = -N22
#     LMI31 = zeros(iμ,ix)
#     LMI32 = zeros(iμ,iψ)
#     LMI33 = -N11
#     LMI = [LMI11 LMI21' LMI31';
#         LMI21 LMI22 LMI32';
#         LMI31 LMI32 LMI33
#     ]
#     dHdb = vec(LMI)
#     Sb .= dfg .* dλdH * dHdb

#     return Aq,Sy,Sz,Sb
# end

# function diff_numeric(model::FunnelCTCS,q::Vector, y::Vector,z::Vector,b::Float64)
#     iq = size(q,1)
#     iy = size(y,1)
#     iz = size(z,1)

#     Aq = zeros(model.is,iq)
#     Sy = zeros(model.is,iy)
#     Sz = zeros(model.is,iq)
#     Sb = zeros(model.is,1)

#     h = 1e-8
#     eps_q = Matrix{Float64}(I,iq,iq)
#     for i in 1:iq
#         Aq[:,i] .= (
#         forward(model,q+h*eps_q[:,i],y,z,b) -
#         forward(model,q-h*eps_q[:,i],y,z,b)
#         ) / (2*h)
#     end

#     eps_y = Matrix{Float64}(I,iy,iy)
#     for i in 1:iy
#         Sy[:,i] .= (
#         forward(model,q,y+h*eps_y[:,i],z,b) -
#         forward(model,q,y-h*eps_y[:,i],z,b)
#         ) / (2*h)
#     end

#     eps_z = Matrix{Float64}(I,iz,iz)
#     for i in 1:iz
#         Sz[:,i] .= (
#         forward(model,q,y,z+h*eps_z[:,i],b) -
#         forward(model,q,y,z-h*eps_z[:,i],b)
#         ) / (2*h)
#     end

#     Sb[:,1] .= (
#         forward(model,q,y,z,b+h) -
#         forward(model,q,y,z,b-h)
#         ) / (2*h)
#     return Aq,Sy,Sz,Sb
# end


# function discretize_foh(model::FunnelDynamics,dynamics::Dynamics,ctcs::FunnelCTCS,
#     x::Matrix,u::Matrix,T::Vector,
#     Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3},b::Vector)
#     # Q,Y,Z are unnecessary for linear DLMI, but keep them here for writing nonlinearDLMI later
#     @assert size(x,2) == size(Q,3)
#     @assert size(x,2) + 1 == size(u,2)
#     @assert size(u,2) == size(Y,3)
#     @assert size(Y,3) == size(Z,3)

#     N = size(x,2)
#     ix = model.ix
#     iq = model.iq
#     iy = model.iy
#     is = ctcs.is
#     function get_interval(start,size)
#         return start:(start+size-1)
#     end

#     idx_x = get_interval(1,ix)
#     idx_q = get_interval(idx_x[end]+1,iq)
#     idx_A = get_interval(idx_q[end]+1,iq*iq)
#     idx_Bym = get_interval(idx_A[end]+1,iq*iy)
#     idx_Byp = get_interval(idx_Bym[end]+1,iq*iy)
#     idx_Bzm = get_interval(idx_Byp[end]+1,iq*iq)
#     idx_Bzp = get_interval(idx_Bzm[end]+1,iq*iq)
    
#     idx_s = get_interval(idx_Bzp[end]+1,is)
#     idx_Sq = get_interval(idx_s[end]+1,is*iq)
#     idx_Sym = get_interval(idx_Sq[end]+1,is*iy)
#     idx_Syp = get_interval(idx_Sym[end]+1,is*iy)
#     idx_Szm = get_interval(idx_Syp[end]+1,is*iq)
#     idx_Szp = get_interval(idx_Szm[end]+1,is*iq)
#     idx_Sbm = get_interval(idx_Szp[end]+1,is*1)
#     idx_Sbp = get_interval(idx_Sbm[end]+1,is*1)

#     function dvdt(out,V,p,t)
#         um = p[1]
#         up = p[2]
#         ym = p[3]
#         yp = p[4]
#         zm = p[5]
#         zp = p[6]
#         bm = p[7]
#         bp = p[8]
#         dt = p[9]

#         alpha = (dt - t) / dt
#         beta = t / dt

#         u_ = alpha * um + beta * up
#         y_ = alpha * ym + beta * yp
#         z_ = alpha * zm + beta * zp
#         b_ = alpha * bm + beta * bp

#         x_ = V[idx_x]
#         q_ = V[idx_q]
#         Phi = reshape(V[idx_A], (iq, iq))
#         Bym_ = reshape(V[idx_Bym],(iq,iy))
#         Byp_ = reshape(V[idx_Byp],(iq,iy))
#         Bzm_ = reshape(V[idx_Bzm],(iq,iq))
#         Bzp_ = reshape(V[idx_Bzp],(iq,iq))

#         s_ = V[idx_s]
#         Sq_ = reshape(V[idx_Sq], (is, iq))
#         Sym_ = reshape(V[idx_Sym],(is,iy))
#         Syp_ = reshape(V[idx_Syp],(is,iy))
#         Szm_ = reshape(V[idx_Szm],(is,iq))
#         Szp_ = reshape(V[idx_Szp],(is,iq))
#         Sbm_ = reshape(V[idx_Sbm],(is,1))
#         Sbp_ = reshape(V[idx_Sbp],(is,1))

#         # traj terms
#         f = forward(dynamics,x_,u_)
#         A,B = diff(dynamics,x_,u_)
#         # funl terms
#         F = forward(model,q_,y_,z_,A,B)
#         dAq,dBy,dBz = diff(model,A,B)
#         # ctcs terms
#         ds = forward(ctcs,q_,y_,z_,b_)
#         dSq,dSy,dSz,dSb = diff(ctcs,q_,y_,z_,b_)

#         dxdt = f
#         dqdt = F
#         dAdt = dAq*Phi
#         dBymdt = dAq*Bym_ + dBy.*alpha
#         dBypdt = dAq*Byp_ + dBy.*beta
#         dBzmdt = dAq*Bzm_ + dBz.*alpha
#         dBzpdt = dAq*Bzp_ + dBz.*beta
        
#         dsdt = ds
#         dSqdt = dSq*Phi
#         dSymdt = dSq*Bym_ + dSy.*alpha
#         dSypdt = dSq*Byp_ + dSy.*beta
#         dSzmdt = dSq*Bzm_ + dSz.*alpha
#         dSzpdt = dSq*Bzp_ + dSz.*beta
#         dSbmdt = dSb.*alpha
#         dSbpdt = dSb.*beta

#         dV = [dxdt;dqdt;
#             dAdt[:];dBymdt[:];dBypdt[:];dBzmdt[:];dBzpdt[:];
#             dsdt;dSqdt[:];dSymdt[:];dSypdt[:];dSzmdt[:];dSzpdt[:];
#             dSbmdt[:];dSbpdt[:]]
#         out .= dV[:]
#     end

#     Aq = zeros(iq,iq,N)
#     Bym = zeros(iq,iy,N)
#     Byp = zeros(iq,iy,N)
#     Bzm = zeros(iq,iq,N)
#     Bzp = zeros(iq,iq,N)

#     Sq = zeros(is,iq,N)
#     Sym = zeros(is,iy,N)
#     Syp = zeros(is,iy,N)
#     Szm = zeros(is,iq,N)
#     Szp = zeros(is,iq,N)
#     Sbm = zeros(is,1,N)
#     Sbp = zeros(is,1,N)
#     SZ = zeros(is,N)

#     x_prop = zeros(ix,N)
#     q_prop = zeros(iq,N)
#     s_prop = zeros(is,N)

#     for i = 1:N
#         A0 = I(iq)
#         Bym0 = zeros(iq,iy)
#         Byp0 = zeros(iq,iy)
#         Bzm0 = zeros(iq,iq)
#         Bzp0 = zeros(iq,iq)
#         S0 = zeros(is,iq)
#         Sym0 = zeros(is,iy)
#         Syp0 = zeros(is,iy)
#         Szm0 = zeros(is,iq)
#         Szp0 = zeros(is,iq)
#         Sbm0 = zeros(is,1)
#         Sbp0 = zeros(is,1)
#         V0 = [x[:,i];vec(Q[:,:,i]);A0[:];Bym0[:];Byp0[:];Bzm0[:];Bzp0[:];
#             zeros(is);S0[:];Sym0[:];Syp0[:];Szm0[:];Szp0[:];Sbm0[:];Sbp0[:]][:]

#         um = u[:,i]
#         up = u[:,i+1]
#         ym = vec(Y[:,:,i])
#         yp = vec(Y[:,:,i+1])
#         zm = vec(Z[:,:,i])
#         zp = vec(Z[:,:,i+1])
#         bm = b[i]
#         bp = b[i+1]
#         dt = T[i]

#         t, sol = RK4(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,bm,bp,dt),50)
#         x_prop[:,i] .= sol[idx_x,end]
#         q_prop[:,i] .= sol[idx_q,end]
#         Aq[:,:,i] .= reshape(sol[idx_A,end],iq,iq)
#         Bym[:,:,i] .= reshape(sol[idx_Bym,end],iq,iy)
#         Byp[:,:,i] .= reshape(sol[idx_Byp,end],iq,iy)
#         Bzm[:,:,i] .= reshape(sol[idx_Bzm,end],iq,iq)
#         Bzp[:,:,i] .= reshape(sol[idx_Bzp,end],iq,iq)
#         s_prop[:,i] .= sol[idx_s,end]
#         Sq[:,:,i] .= reshape(sol[idx_Sq,end],is,iq)
#         Sym[:,:,i] .= reshape(sol[idx_Sym,end],is,iy)
#         Syp[:,:,i] .= reshape(sol[idx_Syp,end],is,iy)
#         Szm[:,:,i] .= reshape(sol[idx_Szm,end],is,iq)
#         Szp[:,:,i] .= reshape(sol[idx_Szp,end],is,iq)
#         Sbm[:,:,i] .= reshape(sol[idx_Sbm,end],is,1)
#         Sbp[:,:,i] .= reshape(sol[idx_Sbp,end],is,1)
#         SZ[:,i] .= (s_prop[:,i] - Sq[:,:,i]*vec(Q[:,:,i]) 
#             - Sym[:,:,i]*ym - Syp[:,:,i]*yp
#             - Szm[:,:,i]*zm - Szp[:,:,i]*zp
#             - Sbm[:,:,i]*bm - Sbp[:,:,i]*bp)
#     end
#     return Aq,Bym,Byp,Bzm,Bzp,Sq,Sym,Syp,Szm,Szp,Sbm,Sbp,SZ,x_prop,q_prop,s_prop
# end