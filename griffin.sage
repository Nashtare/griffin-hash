from CompactFIPS202 import SHAKE256


##########################################################
# HELPER FUNCTIONS FOR GENERATING GRIFFIN INSTANTIATIONS #
##########################################################

def get_round_constants(p, t, capacity, security_level, N):
    # generate pseudorandom bytes
    bytes_per_int = ceil(len(bin(p)[2:]) / 8) + 1
    # 1 value for alpha_2
    # 1 value for beta_2
    # t * (N-1) values for ARK
    num_elems = (t * (N - 1) + 2)
    num_bytes = bytes_per_int * num_elems
    seed_string = "Griffin(%i,%i,%i,%i)" % (p, t, capacity, security_level)
    byte_string = SHAKE256(bytes(seed_string, "ascii"), num_bytes)

    # process byte string in chunks
    round_constants = []
    alphas = []
    betas = []
    Fp = FiniteField(p)

    # generate alpha_2 and deduce the other ones
    chunk = byte_string[0: bytes_per_int]
    alpha = Fp(sum(256 ** j * ZZ(chunk[j]) for j in range(len(chunk))))
    alphas.append(alpha)
    for i in range(3, t):
        alphas.append(Fp(i - 1) * alpha)

    # generate beta_2 and deduce the other ones
    chunk = byte_string[bytes_per_int: bytes_per_int*2]
    beta = Fp(sum(256 ** j * ZZ(chunk[j]) for j in range(len(chunk))))
    betas.append(beta)
    for i in range(3, t):
        betas.append(Fp(i - 1)**2 * beta)

    for i in range(2, num_elems):
        chunk = byte_string[bytes_per_int*i: bytes_per_int*(i+1)]
        c = Fp(sum(256 ** j * ZZ(chunk[j]) for j in range(len(chunk))))
        round_constants.append(c)

    return alphas, betas, round_constants


def get_number_of_rounds(p, t, security_level, d):
    assert security_level <= min(256, floor(log(p, 2) * t/3.0))
    # get number of rounds for Groebner basis attack
    target = 2 ** (security_level // 2)
    for rgb in range(1, 25):
        left = binomial(rgb * (d + t) + 1, 1 + t * rgb)
        right = binomial(d**rgb + 1 + rgb, 1 + rgb)
        if min(left, right) >= target:
            break

    # set a minimum value for sanity and add 20%
    return ceil(1.2 * max(6, 1 + rgb))


def get_powers(p):
    for d in range(3, p):
        if gcd(d, p-1) == 1:
            break
    g, dinv, garbage = xgcd(d, p-1)
    return (d, (dinv % (p-1)))


def get_matrix(p, t):
    # TODO: the decomposition below is overly complicated. It
    # would probably be simpler to rely on numpy.
    Fp = FiniteField(p)
    if t == 3:
        return Matrix.circulant([2, 1, 1]).change_ring(Fp)
    if t == 4:
        return Matrix.circulant([3, 2, 1, 1]).change_ring(Fp)

    # for larger cases, we split the matrix M as M' x M''
    # with M' a diagonal matrix and M'' a circulant one.
    # this requires t to be a multiple of 4
    assert t % 4 == 0
    tp = t // 4

    Mt = Matrix.circulant([3, 2, 1, 1]).change_ring(Fp)
    M1 = Matrix.zero(t, t)
    # put Mt on the diagonal of the larger matrix M1
    for i in range(tp):
        for row in range(4):
            for col in range(4):
                M1[4*i + row, 4*i + col] = Mt[row, col]

    M2 = Matrix.diagonal([1 for i in range(t)])
    # we fill up the missing non-zero coefficients so
    # that M2 looks like = circ(2I_4, I_4, ..., I_4).
    # we proceed to do so in two phases as the matrix is
    # symmetric.
    for col in range(1, tp):
        for row in range(0, col):
            for diag in range(4):
                M2[4*row + diag, 4*col + diag] = 1
    # now M2 is upper-triangular, we can transpose and add
    # it to obtain the desired matrix
    M2 = M2 + M2.transpose()

    M = M1 * M2

    return M.change_ring(Fp)


def griffin_parameters(p, t, capacity, security_level):
    d, dinv = get_powers(p)
    N = get_number_of_rounds(p, t, security_level, d)
    mat = get_matrix(p, t)
    alphas, betas, round_constants = get_round_constants(
        p, t, capacity, security_level, N)
    return p, t, capacity, security_level, d, dinv, N, mat, alphas, betas, round_constants


#################################################
# INTERNAL GRIFFIN PERMUTATION HELPER FUNCTIONS #
#################################################

def nonlinear_layer(parameters, state):
    p, t, capacity, security_level, d, dinv, N, mat, alphas, betas, round_constants = parameters
    Fp = FiniteField(p)
    # the original definition takes as input lambda_i, which equals i - 1
    def Li(z0, z1, z2, i): return Fp((i-1) * z0 + z1 + z2)

    state[0, 0] = state[0, 0] ** dinv
    state[1, 0] = state[1, 0] ** d

    l = Li(state[0, 0], state[1, 0], 0, 2)
    state[2, 0] = state[2, 0] * (l ** 2 + alphas[0] * l + betas[0])

    for i in range(3, t):
        l = Li(state[0, 0], state[1, 0], state[i - 1, 0], i)
        state[i, 0] = state[i, 0] * (l ** 2 + alphas[i - 2] * l + betas[i - 2])
    return state


def linear_layer(parameters, state):
    p, t, capacity, security_level, d, dinv, N, mat, alphas, betas, round_constants = parameters
    Fp = FiniteField(p)
    state = mat * state
    return state


def additive_constants_layer(parameters, state, round):
    p, t, capacity, security_level, d, dinv, N, mat, alphas, betas, round_constants = parameters
    Fp = FiniteField(p)
    for j in range(t):
        state[j, 0] += round_constants[round*t + j]
    return state


##########################################
# GRIFFIN HASH AND COMPRESSION FUNCTIONS #
##########################################

def griffin_permutation(parameters, state):
    p, t, capacity, security_level, d, dinv, N, mat, alphas, betas, round_constants = parameters

    for i in range(N - 1):
        state = nonlinear_layer(parameters, state)
        state = linear_layer(parameters, state)
        state = additive_constants_layer(parameters, state, i)

    # for the last round, we do not add any round constants to the state
    state = nonlinear_layer(parameters, state)
    state = linear_layer(parameters, state)

    return state


def griffin_sponge(parameters, input_sequence, output_length):
    p, t, capacity, security_level, d, dinv, N, mat, alphas, betas, round_constants = parameters
    rate = t - capacity
    Fp = FiniteField(p)

    length = len(input_sequence)

    # initialize state to all zeros then conditionally set the first capacity register to one
    state = matrix([[Fp(0)] for i in range(t)])
    if length % rate != 0:
        state[rate, 0] = 1

    if length % rate != 0:
        input_sequence.append(1)

    # we finish the simplest padding rule by appending zeros as necessary
    while len(input_sequence) % rate != 0:
        input_sequence.append(0)

    # absorbing
    absorb_index = 0
    while absorb_index < len(input_sequence):
        for i in range(0, rate):
            state[i, 0] += input_sequence[absorb_index]
            absorb_index += 1
        state = griffin_permutation(parameters, state)

    # squeezing
    output_sequence = []
    squeeze_index = 0
    while squeeze_index < output_length:
        for i in range(0, rate):
            output_sequence.append(state[i, 0])
            squeeze_index += 1
        if squeeze_index < output_length:
            state = griffin_permutation(parameters, state)

    return output_sequence[:output_length]


########################################################

# Griffin instantiation to be used with Jive compression mode

p = 2**64 - 2**32 + 1  # STARK-friendly Goldilocks field
S = 128  # Security level
t = 12  # Number of base field elements
c = 4  # Number of elements dedicated to the capacity
p, t, capacity, security_level, d, dinv, N, mat, alphas, betas, round_constants = griffin_parameters(
    p, t, c, S)
parameters = p, t, capacity, security_level, d, dinv, N, mat, alphas, betas, round_constants

# Test vectors
Fp = GF(p)
L = [[0 for i in range(8)], [1 for i in range(8)]]
for j in range(10):
    L.append([Fp.random_element() for i in range(8)])
print(L)

R = []
for l in L:
    R.append(griffin_sponge(parameters, l, 4))
print(R)
