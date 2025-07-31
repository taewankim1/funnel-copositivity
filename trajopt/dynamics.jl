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
    ilam::Int
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
        iq = 6
        iphi = 2
        ir = 2 + 2 + 2 + iq
        ip = 6 + iphi
        idelta = 6 # What is this for?
        @assert iq == idelta
        ilam =  idelta + iphi # will be initialized later.

        Co1 = [0 0 1; 0 0 0; 0 0 0]
        Do1 = [0 0; 1 0; 0 0]
        Go1 = [0 0; 0 0; 1 0]
        Co = [Co1;Co1]
        Do = [Do1;Do1]
        Go = [Go1;Go1]
        Eo = [1 0; 0 1; 0 0]

        c1 = 0.03
        c2 = 0.05
        new(ix, iu, iw, iq, iphi, ir, ip, idelta, ilam, Co, Do, Eo, Go, c1, c2)
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

mutable struct Rocket <: Dynamics
    ix::Int64
    iu::Int64
    iw::Int64

    m::Float64
    J_x::Float64
    J_y::Float64
    J_z::Float64

    r_t::Float64
    g::Float64

    # Selector matrices.
    type_channel::Int64
    iq::Int
    iq_list::Array{Int64}
    iphi::Int
    ir::Int
    ip::Int
    idelta::Int
    ilam::Int
    Co::Array{Float64,2}
    Do::Array{Float64,2}
    Eo::Array{Float64,2}
    

    # Lipschitz and Lsmooth contants.
    gamma::Array{Float64,3}
    beta::Vector{Float64}

    # Will be initialized later.
    C::Array{Float64,2}
    D::Array{Float64,2}
    E::Array{Float64,2}
    function Rocket(;type_channel::Int64=1)
        m0 = 2
        J_x = 0.29292
        J_y = 0.29292
        J_z = 0.0025

        r_t = 0.25
        g = 1.625

        # Here, we consider nonlinearity and approximation error simultaneosuly.
        # That is, considering large enough beta can cover uncertainty for the incremental system,
        # caused by nonlinearity. Hence, 'iq' and 'iphi' are set to zeros.
        iq = 0
        iphi = 0

        if type_channel == 1
            iq_list = [6 6 5 4 3 4 2 2 2]
            C1 = [zeros(6,6) [Matrix(1.0I,3,3);zeros(3,3)] zeros(6,3)]
            D1 = [zeros(3,6); Matrix(1.0I,3,3) zeros(3,3)]
            C2 = copy(C1)
            D2 = copy(D1)
            C3 = [zeros(5,6) [Matrix(1.0I,2,3);zeros(3,3)] zeros(5,3)]
            D3 = [zeros(2,6); Matrix(1.0I,3,3) zeros(3,3)]
            C4 = [zeros(4,6) [1 0 0 0 0 0;0 1 0 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0]]
            D4 = zeros(4,6)
            C5 = [zeros(3,6) [1 0 0 0 0 0;0 0 0 1 0 0; 0 0 0 0 1 0]]
            D5 = zeros(3,6)
            C6 = copy(C4)
            D6 = copy(D4)
            C7 = [zeros(2,9) [0 1 0;0 0 1]]
            D7 = zeros(2,6)
            C8 = [zeros(2,9) [1 0 0;0 0 1]]
            D8 = zeros(2,6)
            C9 = [zeros(2,9) [1 0 0;0 1 0]]
            D9 = zeros(2,6)
            Co = [C1;C2;C3;C4;C5;C6;C7;C8;C9]
            Do = [D1;D2;D3;D4;D5;D6;D7;D8;D9]
            Eo = [zeros(3,9);Matrix(1.0I,9,9)] 
            idelta = size(Eo,2)
            ir = size(Co,1)
            ip = idelta
            ilam = ip
        elseif type_channel == 2
            iq_list = [6 4 3]
            C1 = [zeros(6,6) [Matrix(1.0I,3,3);zeros(3,3)] zeros(6,3)]
            D1 = [zeros(3,6); Matrix(1.0I,3,3) zeros(3,3)]
            C2 = [zeros(4,6) [1 0 0 0 0 0;0 1 0 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0]]
            D2 = zeros(4,6)
            C3 = [zeros(3,9) [1 0 0;0 1 0;0 0 1]]
            D3 = zeros(3,6)
            Co = [C1;C2;C3]
            Do = [D1;D2;D3]
            Eo = [zeros(3,9);Matrix(1.0I,9,9)] 
            idelta = 3
            ir = size(Co,1)
            ip = size(Eo,2)
            ilam = 3
        end

        new(12,6,0,m0,J_x,J_y,J_z,r_t,g,type_channel,iq,iq_list,iphi,ir,ip,idelta,ilam,Co,Do,Eo)
    end
end

function forward(model::Rocket,x::Vector,u::Vector)
    rx = x[1]
    ry = x[2]
    rz = x[3]
    vx = x[4]
    vy = x[5]
    vz = x[6]
    phi = x[7]
    theta = x[8]
    psi = x[9]
    p = x[10]
    q = x[11]
    r = x[12]

    Fx = u[1]
    Fy = u[2]
    Fz = u[3]
    Tx = u[4]
    Ty = u[5]
    Tz = u[6]

    # alpha_m = model.alpha_ME
    # alpha_r = model.alpha_RCS
    m = model.m
    J_x = model.J_x
    J_y = model.J_y
    J_z = model.J_z

    r_t = model.r_t
    g = model.g

    f = zeros(size(x))
    f[1] = vx
    f[2] = vy
    f[3] = vz
    f[4] = Fx*cos(psi)*cos(theta)/m + Fy*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))/m + Fz*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))/m
    f[5] = Fx*sin(psi)*cos(theta)/m + Fy*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))/m + Fz*(-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))/m
    f[6] = -Fx*sin(theta)/m + Fy*sin(phi)*cos(theta)/m + Fz*cos(phi)*cos(theta)/m - g
    f[7] = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
    f[8] = q*cos(phi) - r*sin(phi)
    f[9] = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)
    f[10] = (Fy*r_t + J_y*q*r - J_z*q*r + Tx)/J_x
    f[11] = (-Fx*r_t - J_x*p*r + J_z*p*r + Ty)/J_y
    f[12] = (J_x*p*q - J_y*p*q + Tz)/J_z
    return f
end

function diff(model::Rocket, x::Vector, u::Vector)
    rx = x[1]
    ry = x[2]
    rz = x[3]
    vx = x[4]
    vy = x[5]
    vz = x[6]
    phi = x[7]
    theta = x[8]
    psi = x[9]
    p = x[10]
    q = x[11]
    r = x[12]

    Fx = u[1]
    Fy = u[2]
    Fz = u[3]
    Tx = u[4]
    Ty = u[5]
    Tz = u[6]

    # alpha_m = model.alpha_ME
    # alpha_r = model.alpha_RCS
    m = model.m
    J_x = model.J_x
    J_y = model.J_y
    J_z = model.J_z

    r_t = model.r_t
    g = model.g

    fx = zeros(model.ix,model.ix)
    fx[1,4] = 1
    fx[2,5] = 1
    fx[3,6] = 1
    fx[4,7] = Fy*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))/m + Fz*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))/m
    fx[4,8] = -Fx*sin(theta)*cos(psi)/m + Fy*sin(phi)*cos(psi)*cos(theta)/m + Fz*cos(phi)*cos(psi)*cos(theta)/m
    fx[4,9] = -Fx*sin(psi)*cos(theta)/m + Fy*(-sin(phi)*sin(psi)*sin(theta) - cos(phi)*cos(psi))/m + Fz*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi))/m
    fx[5,7] = Fy*(-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))/m + Fz*(-sin(phi)*sin(psi)*sin(theta) - cos(phi)*cos(psi))/m
    fx[5,8] = -Fx*sin(psi)*sin(theta)/m + Fy*sin(phi)*sin(psi)*cos(theta)/m + Fz*sin(psi)*cos(phi)*cos(theta)/m
    fx[5,9] = Fx*cos(psi)*cos(theta)/m + Fy*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))/m + Fz*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))/m
    fx[6,7] = Fy*cos(phi)*cos(theta)/m - Fz*sin(phi)*cos(theta)/m
    fx[6,8] = -Fx*cos(theta)/m - Fy*sin(phi)*sin(theta)/m - Fz*sin(theta)*cos(phi)/m
    fx[7,7] = q*cos(phi)*tan(theta) - r*sin(phi)*tan(theta)
    fx[7,8] = q*(tan(theta)^2 + 1)*sin(phi) + r*(tan(theta)^2 + 1)*cos(phi)
    fx[7,10] = 1
    fx[7,11] = sin(phi)*tan(theta)
    fx[7,12] = cos(phi)*tan(theta)
    fx[8,7] = -q*sin(phi) - r*cos(phi)
    fx[8,11] = cos(phi)
    fx[8,12] = -sin(phi)
    fx[9,7] = q*cos(phi)/cos(theta) - r*sin(phi)/cos(theta)
    fx[9,8] = q*sin(phi)*sin(theta)/cos(theta)^2 + r*sin(theta)*cos(phi)/cos(theta)^2
    fx[9,11] = sin(phi)/cos(theta)
    fx[9,12] = cos(phi)/cos(theta)
    fx[10,11] = (J_y*r - J_z*r)/J_x
    fx[10,12] = (J_y*q - J_z*q)/J_x
    fx[11,10] = (-J_x*r + J_z*r)/J_y
    fx[11,12] = (-J_x*p + J_z*p)/J_y
    fx[12,10] = (J_x*q - J_y*q)/J_z
    fx[12,11] = (J_x*p - J_y*p)/J_z

    fu = zeros(model.ix,model.iu)
    fu[4,1] = cos(psi)*cos(theta)/m
    fu[4,2] = (sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))/m
    fu[4,3] = (sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))/m
    fu[5,1] = sin(psi)*cos(theta)/m
    fu[5,2] = (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))/m
    fu[5,3] = (-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))/m
    fu[6,1] = -sin(theta)/m
    fu[6,2] = sin(phi)*cos(theta)/m
    fu[6,3] = cos(phi)*cos(theta)/m
    fu[10,2] = r_t/J_x
    fu[10,4] = 1/J_x
    fu[11,1] = -r_t/J_y
    fu[11,5] = 1/J_y
    fu[12,6] = 1/J_z
    return fx, fu
end