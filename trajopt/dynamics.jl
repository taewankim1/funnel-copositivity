abstract type Dynamics end

mutable struct Unicycle <: Dynamics
    ix::Int
    iu::Int
    iw::Int
    iq::Int
    iphi::Int
    ir::Int
    ip::Int
    idelta::Int
    Co::Array{Float64,2}
    Do::Array{Float64,2}
    Eo::Array{Float64,2}
    Go::Array{Float64,2}
    c1::Float64
    c2::Float64

    # will be initialized later
    C::Array{Float64,2}
    D::Array{Float64,2}
    E::Array{Float64,2}
    G::Array{Float64,2}
    function Unicycle()
        ix = 3
        iu = 2
        iw = 2
        iq = 3
        iphi = 2
        ir = ix + iu + iw + iq
        ip = ix + iphi
        idelta = 2

        Co = [0 0 1; 0 0 0; 0 0 0]
        Do = [0 0; 1 0; 0 0]
        Eo = [1 0; 0 1; 0 0]
        Go = [0 0; 0 0; 1 0]

        c1 = 0.03
        c2 = 0.05
        new(ix, iu, iw, iq, iphi, ir, ip, idelta, Co, Do, Eo, Go, c1, c2)
    end
end

function forward(model::Unicycle, x::Vector, u::Vector)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]

    v = u[1]
    w = u[2]

    f = zeros(size(x))
    f[1] = v * cos(x3)
    f[2] = v * sin(x3)
    f[3] = w
    return f
end

function forward_uncertain(model::Unicycle, x::Vector, u::Vector, w::Vector)::Vector
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]

    u1 = u[1]
    u2 = u[2]

    w1 = w[1]
    w2 = w[2]

    c1 = model.c1
    c2 = model.c2

    f = zeros(size(x))
    f[1] = u1 * cos(x3 + c1 * w1)
    f[2] = u1 * sin(x3 + c1 * w1)
    f[3] = u2 + c2 * w2
    return f
end

function diff(model::Unicycle, x::Vector, u::Vector)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]

    v = u[1]
    w = u[2]

    fx = zeros(model.ix, model.ix)
    fx[1, 1] = 0.0
    fx[1, 2] = 0.0
    fx[1, 3] = -v * sin(x3)
    fx[2, 1] = 0.0
    fx[2, 2] = 0.0
    fx[2, 3] = v * cos(x3)
    fx[3, 1] = 0.0
    fx[3, 2] = 0.0
    fx[3, 3] = 0.0
    fu = zeros(model.ix, model.iu)
    fu[1, 1] = cos(x3)
    fu[1, 2] = 0.0
    fu[2, 1] = sin(x3)
    fu[2, 2] = 0.0
    fu[3, 1] = 0.0
    fu[3, 2] = 1.0
    return fx, fu
end

function diff_ABF(model::Unicycle, x::Vector, u::Vector)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    u1 = u[1]
    u2 = u[2]
    # w1 = w[1]
    # w2 = w[2]
    c1 = model.c1
    c2 = model.c2

    fx, fu = diff(model, x, u)
    fw = zeros(model.ix, 2)
    fw[1, 1] = -c1 * u1 * sin(x3)
    fw[1, 2] = 0.0
    fw[2, 1] = c1 * u1 * cos(x3)
    fw[2, 2] = 0.0
    fw[3, 1] = 0.0
    fw[3, 2] = c2
    return fx, fu, fw
end


mutable struct QuadrotorDynamics <: Dynamics
    ix::Int
    iu::Int
    iϕ::Int
    iv::Int
    Cv::Array{Float64,2}
    Dvu::Array{Float64,2}
    Go::Union{Vector,Matrix}
    β::Vector{Float64}

    m::Float64
    g::Float64
    Jx::Float64
    Jy::Float64
    Jz::Float64
    function QuadrotorDynamics()
        ix = 12
        iu = 4

        iϕ = 9
        iv = 7
        Go = zeros(ix, iϕ)
        Go[4:end, 1:end] = Matrix(1.0I, iϕ, iϕ)
        Cv = zeros(iv, ix)
        Cv[1, 7] = 1.0
        Cv[2, 8] = 1.0
        Cv[3, 9] = 1.0
        Cv[4, 10] = 1.0
        Cv[5, 11] = 1.0
        Cv[6, 12] = 1.0
        Dvu = zeros(iv, iu)
        Dvu[7, 1] = 1.0
        β = zeros(iϕ)

        m = 1.325
        Jx = 0.03843
        Jy = 0.02719
        Jz = 0.060528
        g = 9.81
        new(ix, iu, iϕ, iv, Cv, Dvu, Go, β, m, g, Jx, Jy, Jz)
    end
end

function forward(model::QuadrotorDynamics, x::Vector, u::Vector)
    # rx = x[1]
    # ry = x[2]
    # rz = x[3]
    vx = x[4]
    vy = x[5]
    vz = x[6]

    phi = x[7]
    theta = x[8]
    psi = x[9]

    p = x[10]
    q = x[11]
    r = x[12]

    Fz = u[1]
    Mx = u[2]
    My = u[3]
    Mz = u[4]

    m = model.m
    g = model.g
    J_x = model.Jx
    J_y = model.Jy
    J_z = model.Jz

    f = zeros(size(x))
    f[1] = vx
    f[2] = vy
    f[3] = vz
    f[4] = Fz * (sin(phi) * sin(psi) + sin(theta) * cos(phi) * cos(psi)) / m
    f[5] = Fz * (-sin(phi) * cos(psi) + sin(psi) * sin(theta) * cos(phi)) / m
    f[6] = Fz * cos(phi) * cos(theta) / m - g
    f[7] = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
    f[8] = q * cos(phi) - r * sin(phi)
    f[9] = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)
    f[10] = (J_y * q * r - J_z * q * r + Mx) / J_x
    f[11] = (-J_x * p * r + J_z * p * r + My) / J_y
    f[12] = (J_x * p * q - J_y * p * q + Mz) / J_z
    return f
end

mutable struct QuadrotorDynamicsNED <: Dynamics
    ix::Int
    iu::Int
    # iϕ::Int
    # iv::Int
    # iψ::Int
    # iμ::Int
    # Cv::Array{Float64,2}
    # Dvu::Array{Float64,2}
    # G::Union{Vector,Matrix}
    # Cμ::Array{Float64,2}
    # Dμu::Array{Float64,2}
    # β::Vector{Float64}

    m::Float64
    g::Float64
    Jx::Float64
    Jy::Float64
    Jz::Float64
    function QuadrotorDynamicsNED()
        ix = 12
        iu = 4
        # iϕ = 6
        # iv = 7

        # iψ = iϕ
        # iμ = iv

        # Go = [0 0 0;0 0 0;0 0 0;1 0 0;0 1 0;0 0 1]
        # Cv = Matrix(1.0I,ix,ix)
        # Dvu = zeros(iu,iu)

        # G = Go
        # Cμ = Cv
        # Dμu = Dvu

        # β = zeros(iψ)

        m = 1.0
        Jx = 1.0
        Jy = 1.0
        Jz = 1.0
        g = 9.81
        new(ix, iu, m, g, Jx, Jy, Jz)
    end
end

function forward(model::QuadrotorDynamicsNED, x::Vector, u::Vector)
    # rx = x[1]
    # ry = x[2]
    # rz = x[3]
    vx = x[4]
    vy = x[5]
    vz = x[6]

    phi = x[7]
    theta = x[8]
    psi = x[9]

    p = x[10]
    q = x[11]
    r = x[12]

    Fz = u[1]
    Mx = u[2]
    My = u[3]
    Mz = u[4]

    m = model.m
    g = model.g
    J_x = model.Jx
    J_y = model.Jy
    J_z = model.Jz

    f = zeros(size(x))
    f[1] = vx
    f[2] = vy
    f[3] = vz
    f[4] = -Fz * (sin(phi) * sin(psi) + sin(theta) * cos(phi) * cos(psi)) / m
    f[5] = -Fz * (-sin(phi) * cos(psi) + sin(psi) * sin(theta) * cos(phi)) / m
    f[6] = -Fz * cos(phi) * cos(theta) / m + g
    f[7] = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
    f[8] = q * cos(phi) - r * sin(phi)
    f[9] = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)
    f[10] = (J_y * q * r - J_z * q * r + Mx) / J_x
    f[11] = (-J_x * p * r + J_z * p * r + My) / J_y
    f[12] = (J_x * p * q - J_y * p * q + Mz) / J_z
    return f
end