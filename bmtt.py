# BOLTZMANN MACHINE THROUGH TIME (name subject to change)
import theano
import theano.tensor as T

Tsigmoid = T.nnet.sigmoid

# Mu.shape = (n_chains, length, dim_h)
# X.shape = (n_chains, length, dim_x)
# W.shape = (dim_t, dim_t+1)
# V.shape = (dim_h, dim_x)
# b.shape = ('x', 'x', dim_h)
# d.shape = ('x', 'x', dim_x)
# Lambda = ('x', 'x', dim_x)
# b_L.shape = b_0.shape = ('x', dim_h)

# P.shape = (n_chains, length, n_phn)
# b_p.shape = ('x', 'x', n_phn)
# U.shape = (dim_h, n_phn)

# MEAN-FIELD UPDATE

# get the mean-field updates of odd timestamps
def update_odd_mu(X, P, Mu, W, V, U, b, b_L):
    l = Mu.shape[1]
    Mu_update = Mu
    
    Mu_update = T.set_subtensor(Mu_update[:,1:-1:2], \
                        Tsigmoid(- T.tensordot(Mu[:,0:-2:2], W, axes = (2,0)) \
                        - T.tensordot(Mu[:,2::2], W, axes = (2,1)) \
                        - T.tensordot(X[:,1:-1:2], V, axes = (2,1)) \
                        - T.tensordot(P[:,1:-1:2], U, axes = (2,1)) \
                        - b))
    Mu_update_last = T.set_subtensor(Mu_update[:,-1], \
                        Tsigmoid(- T.tensordot(Mu[:,-2], W, axes = (1,0)) \
                        - T.tensordot(X[:,-1], V, axes = (1,1)) \
                        - T.tensordot(P[:,-1], U, axes = (1,1)) \
                        - b_L))
    Mu_update = T.switch(T.eq(T.mod(l,2),0), Mu_update_last, Mu_update)
    
    return Mu_update[:,1::2]

# get the mean-field updates of even timestamps
def update_even_mu(X, P, Mu, W, V, U, b, b_0, b_L):
    l = Mu.shape[1]
    Mu_update = Mu
    
    Mu_update = T.set_subtensor(Mu_update[:,2:(l-1):2], \
                        Tsigmoid(- T.tensordot(Mu[:,1:-2:2], W, axes = (2,0)) \
                        - T.tensordot(Mu[:,3::2], W, axes = (2,1)) \
                        - T.tensordot(X[:,2:-1:2], V, axes = (2,1)) \
                        - T.tensordot(P[:,2:-1:2], U, axes = (2,1)) \
                        - b))
    Mu_update = T.set_subtensor(Mu_update[:,0], \
                        Tsigmoid(- T.tensordot(Mu[:,1], W, axes = (1,1)) \
                        - T.tensordot(X[:,0], V, axes = (1,1)) \
                        - T.tensordot(P[:,0], U, axes = (1,1)) \
                        - b_0))
    Mu_update_last = T.set_subtensor(Mu_update[:,-1], \
                        Tsigmoid(- T.tensordot(Mu[:,-2], W, axes = (1,0)) \
                        - T.tensordot(X[:,-1], V, axes = (1,1)) \
                        - T.tensordot(P[:,-1], U, axes = (1,1)) \
                        - b_L))
    Mu_update = T.switch(T.eq(T.mod(l,2),0), Mu_update, Mu_update_last)
    
    return Mu_update[:,::2]

# get the mean-field updates (starting arbitrarily at odd timestamps)
def update_mu(X, P, Mu, W, V, U, b, b_0, b_L):
    Mu_update = T.set_subtensor(Mu[:,1::2], update_odd_mu(X, P, Mu, \
                                W, V, U, b, b_L))
    Mu_update = T.set_subtensor(Mu_update[:,::2], \
                                update_even_mu(X, P, Mu_update, \
                                W, V, U, b, b_0, b_L))
    
    return Mu_update

# get mean-field inference
def get_mean_field_expression(trng, X, P, W, V, U, b, b_0, b_L, n_mf_steps):
    l = X.shape[1]
    
    Mu_init = trng.uniform(size = (1, l, W.shape[0]))
    
    Mus, updates = theano.scan(lambda Mu:update_mu(X, P, Mu, W, V, \
                                                   U, b, b_0, b_L), \
                                outputs_info = Mu_init, \
                                n_steps = n_mf_steps)
    Mu_fin = Mus[-1]
    
    return Mu_fin


# GIBBS SAMPLING 
## get the conditional statistics of X
def get_cmean_x_odd(Mu, V, d, Lambda):
    cmean_x_odd = d - T.tensordot(Mu[:,1::2], V, axes = (2,0))/(T.sqrt(Lambda))
    
    return cmean_x_odd

def get_cmean_x_even(Mu, V, d, Lambda):
    cmean_x_even = d - T.tensordot(Mu[:,::2], V, axes = (2,0))/(T.sqrt(Lambda))
    
    return cmean_x_even

## get the conditional statistics of P
def get_cmean_p_odd(Mu, U, b_p):
    Mu_flatten = Mu[:,1::2].dimshuffle((2,0,1)).flatten(2).dimshuffle((1,0))
    cmean_p_odd = T.nnet.softmax(b_p.flatten(1).dimshuffle(('x',0)) \
                + T.dot(Mu_flatten, U))
    inter_shape = (Mu[:,1::2].shape[0], Mu[:,1::2].shape[1], \
                    b_p.shape[2])
    cmean_p_odd = cmean_p_odd.reshape(inter_shape)
    
    return cmean_p_odd

def get_cmean_p_even(Mu, U, b_p):
    Mu_flatten = Mu[:,::2].dimshuffle((2,0,1)).flatten(2).dimshuffle((1,0))
    cmean_p_even = T.nnet.softmax(b_p.flatten(1).dimshuffle(('x',0)) \
                 + T.dot(Mu_flatten, U))
    inter_shape = (Mu[:,::2].shape[0], Mu[:,::2].shape[1], \
                    b_p.shape[2])
    cmean_p_even = cmean_p_even.reshape(inter_shape)
    
    return cmean_p_even


## resampling

# get the odd resampled H
def get_resampled_odd_h(trng, H, X, P, W, V, U, b, b_L):
    Mu_odd = update_odd_mu(X, P, H, W, V, U, b, b_L)
    H_odd_resampled = trng.binomial(size=Mu_odd.shape, p=Mu_odd)
    
    return H_odd_resampled

# get the even resampled H
def get_resampled_even_h(trng, H, X, P, W, V, U, b, b_0, b_L):
    Mu_even = update_even_mu(X, P, H, W, V, U, b, b_0, b_L)
    H_even_resampled = trng.binomial(size=Mu_even.shape, p=Mu_even)
    
    return H_even_resampled

# get the odd resampled X
def get_resampled_odd_x(trng, Mu, V, d, Lambda):
    cmean_x_odd = get_cmean_x_odd(Mu, V, d, Lambda)
    x_odd_resampled = cmean_x_odd \
            + trng.normal(size=cmean_x_odd.shape)/(T.sqrt(Lambda))
    
    return x_odd_resampled

# get the even resampled X
def get_resampled_even_x(trng, Mu, V, d, Lambda):
    cmean_x_even = get_cmean_x_even(Mu, V, d, Lambda)
    x_even_resampled = cmean_x_even \
            + trng.normal(size=cmean_x_even.shape)/(T.sqrt(Lambda))
    
    return x_even_resampled

# get the odd resampled P
def get_resampled_odd_p(trng, Mu, U, b_p):
    cmean_p_odd = get_cmean_p_odd(Mu, U, b_p)
    p_odd_resampled = trng.multinomial(pvals=cmean_p_odd)
    
    return p_odd_resampled

# get the even resampled P
def get_resampled_even_p(trng, Mu, U, b_p):
    cmean_p_even = get_cmean_p_even(Mu, U, b_p)
    p_even_resampled = trng.multinomial(pvals=cmean_p_even)
    
    return p_even_resampled


## Gibbs sampling

# get resamples of a subset of H
def get_gibbs_h(trng, H, X, P, W, V, U, b, b_0, b_L, subset):
    assert subset in ["even", "odd"]
    
    if subset == "odd":
        H_resampled = T.set_subtensor(H[:,1::2], \
                        get_resampled_odd_h(trng, H, X, P, W, V, U, b, b_L))
    else:
        H_resampled = T.set_subtensor(H[:,::2], \
                        get_resampled_even_h(trng, H, X, P, W, V, U, b, \
                                            b_0, b_L))
    
    return H_resampled

# get resamples of a subset of X
def get_gibbs_x(trng, H, X, V, d, Lambda, subset):
    assert subset in ["even", "odd"]
    
    if subset == "odd":
        X_resampled = T.set_subtensor(X[:,1::2], \
                        get_resampled_odd_x(trng, H, V, d, Lambda))
    else:
        X_resampled = T.set_subtensor(X[:,::2], \
                        get_resampled_even_x(trng, H, V, d, Lambda))
    
    return X_resampled

# get resamples of a subset of P
def get_gibbs_p(trng, H, P, U, b_p, subset):
    assert subset in ["even", "odd"]
    
    if subset == "odd":
        P_resampled = T.set_subtensor(P[:,1::2], \
                        get_resampled_odd_p(trng, H, U, b_p))
    else:
        P_resampled = T.set_subtensor(P[:,::2], \
                        get_resampled_even_p(trng, H, U, b_p))
    
    return P_resampled


# get the a step of Gibbs sampling
def get_gibbs_global_one_step(trng, H, X, P, W, V, U, b, b_0, b_L, d, \
                            Lambda, b_p, subset_h):
    assert subset_h in ["even", "odd"]
    
    if subset_h == "odd":
        H_resampled = get_gibbs_h(trng, H, X, P, W, V, U, b, b_0, b_L, "odd")
        X_resampled = get_gibbs_x(trng, H_resampled, X, V, d, Lambda, "even")
        P_resampled = get_gibbs_p(trng, H_resampled, P, U, b_p, "even")
    else:
        H_resampled = get_gibbs_h(trng, H, X, P, W, V, U, b, b_0, b_L, "even")
        X_resampled = get_gibbs_x(trng, H_resampled, X, V, d, Lambda, "odd")
        P_resampled = get_gibbs_p(trng, H_resampled, P, U, b_p, "odd")
    
    return [X_resampled, P_resampled, H_resampled]

# get two steps of Gibbs sampling
def get_gibbs_global_two_step(trng, H, X, P, W, V, U, b, b_0, b_L, d, \
                            Lambda, b_p, first_subset_h, clamp_x, clamp_p):
    
    assert first_subset_h in ["even", "odd"]
    
    if first_subset_h == "odd":
        X_resampled, P_resampled, H_resampled = \
                                get_gibbs_global_one_step(trng, H, X, P, W, \
                                V, U, b, b_0, b_L, d, Lambda, b_p, "odd")
    else:
        X_resampled, P_resampled, H_resampled = \
                                get_gibbs_global_one_step(trng, H, X, P, W, \
                                V, U, b, b_0, b_L, d, Lambda, b_p, "even")
    
    if clamp_x:
        X_resampled = X
    if clamp_p:
        P_resampled = P
    
    if first_subset_h == "odd":
        X_resampled, P_resampled, H_resampled = \
                get_gibbs_global_one_step(trng, H_resampled, X_resampled, \
                P_resampled, W, V, U, b, b_0, b_L, d, Lambda, b_p,  "even")
    else:
        X_resampled, P_resampled, H_resampled = \
                get_gibbs_global_one_step(trng, H_resampled, X_resampled, \
                P_resampled, W, V, U, b, b_0, b_L, d, Lambda, b_p,  "odd")
    
    if clamp_x:
        X_resampled = X
    if clamp_p:
        P_resampled = P
    
    return [X_resampled, P_resampled, H_resampled]

# get CD-k fantasy particles
def get_fantasy_particles_expressions(trng, X, P, W, V, U, b, b_0, b_L, \
                        d, Lambda, b_p, n_samples, n_gibbs_step, phase, \
                        subset_h_start, Mu = None, use_mode = False):
    assert phase in ["positive", "negative"]
    assert subset_h_start in ["even", "odd"]
    
    clamp_x = (phase == "positive")
    clamp_p = clamp_x
    
    l = X.shape[1]
    X_init = T.tensordot(T.ones((n_samples,1)), X, axes=(1,0))
    P_init = T.tensordot(T.ones((n_samples,1)), P, axes=(1,0))
    
    if Mu is None:
        H_init = trng.binomial(size = (n_samples, l, W.shape[0]))
    else:
        H_init = trng.binomial(size = (n_samples, l, W.shape[0]), p = Mu)
    
    ([Xs, Ps, Hs], updates) = theano.scan(lambda X_gibbs, P_gibbs, H: \
                get_gibbs_global_two_step(trng, H, X_gibbs, P_gibbs, \
                                W, V, U, b, b_0, b_L, d, Lambda, b_p, \
                                subset_h_start, clamp_x, clamp_p), \
                outputs_info = [X_init, P_init, H_init], \
                n_steps = n_gibbs_step)
    
    X_fin = Xs[-1]
    P_fin = Ps[-1]
    H_fin = Hs[-1]
    
    if use_mode:
        X_fin = T.set_subtensor(X_fin[:,1::2], get_cmean_x_odd(H_fin, V, d, Lambda))
        X_fin = T.set_subtensor(X_fin[:,::2], get_cmean_x_even(H_fin, V, d, Lambda))
        
        #P_fin = T.set_subtensor(P_fin[:,1::2], get_cmean_p_odd(H_fin, U, b_p))
        #P_fin = T.set_subtensor(P_fin[:,::2], get_cmean_p_even(H_fin, U, b_p))
        
        #P_fin = T.set_subtensor(T.zeros_like(P_fin)[T.argmax(P_fin, 2)], 1)
    
    return X_fin, P_fin, H_fin, updates


# COMPUTING ENERGY

# compute the energy with optional Rao-Blackwellization
def get_energy(X, P, H, W, V, U, b, b_0, b_L, d, Lambda, b_p, \
               x_marginalized = None, p_marginalized = None):
    assert x_marginalized in ["even", "odd", None]
    
    # energy term for the h pair factors
    energy_factors_hh = (T.tensordot(H[:,:-1], W, \
                axes = (2,0))*H[:,1:]).sum(axis=2).sum(axis=1).mean()
    
    # energy term for the individual h factors
    energy_factors_h_mid = (H[:,1:-1]*b).sum(axis=2).sum(axis=1).mean()
    energy_factors_h_0 = (H[:,0]*b_0).sum(axis=1).mean()
    energy_factors_h_L = (H[:,-1]*b_L).sum(axis=1).mean()
    
    
    if x_marginalized == "even":
        # marginalize even X
        
        ## energy term for the x factors
        cmean_x_even = get_cmean_x_even(H, V, d, Lambda)
        mmd = cmean_x_even - d
        
        energy_factors_x_even = \
            0.5*((Lambda*(mmd**2)).sum(axis=2).sum(axis=1).mean() \
            + cmean_x_even.shape[1]*X.shape[2])
        
        xmd = X - d
        energy_factors_x_odd = \
            0.5*(Lambda*(xmd[:,1::2]**2)).sum(axis=2).sum(axis=1).mean()
        
        energy_factors_x = energy_factors_x_odd + energy_factors_x_even
        
        ## energy term for the x-h pair factors
        Xm_even = T.set_subtensor(X[:,::2], cmean_x_even)
        energy_factors_xh = (T.tensordot(H, V, \
                axes = (2,0))*Xm_even).sum(axis=2).sum(axis=1).mean()
        
    elif x_marginalized == "odd":
        # marginalize odd X
        
        ## energy term for the x factors
        cmean_x_odd = get_cmean_x_odd(H, V, d, Lambda)
        mmd = cmean_x_odd - d
        
        energy_factors_x_odd = \
            0.5*((Lambda*(mmd**2)).sum(axis=2).sum(axis=1).mean() \
            + cmean_x_odd.shape[1]*X.shape[2])
        
        xmd = X - d
        energy_factors_x_even = \
            0.5*(Lambda*(xmd[:,::2]**2)).sum(axis=2).sum(axis=1).mean()
        
        energy_factors_x = energy_factors_x_odd + energy_factors_x_even
        
        ## energy term for the x-h pair factors
        Xm_odd = T.set_subtensor(X[:,1::2], cmean_x_odd)
        energy_factors_xh = (T.tensordot(H, V, \
                axes = (2,0))*Xm_odd).sum(axis=2).sum(axis=1).mean()
        
    else:
        xmd = X - d
        energy_factors_x = 0.5*(Lambda*(xmd**2)).sum(axis=2).sum(axis=1).mean()
        
        # energy term for the x-h pair factors
        energy_factors_xh = (T.tensordot(H, V, \
                    axes = (2,0))*X).sum(axis=2).sum(axis=1).mean()
    
    # energy term for the p-h pair factors
    if p_marginalized == "even":
        # marginalize even P
        cmean_p_even = get_cmean_p_even(H, U, b_p)
        Pm = T.set_subtensor(P[:,::2], cmean_p_even)
    elif p_marginalized == "odd":
        # marginalize odd P
        cmean_p_odd = get_cmean_p_odd(H, U, b_p)
        Pm = T.set_subtensor(P[:,1::2], cmean_p_odd)
    else:
        Pm = P
    
    energy_factor_ph = (T.tensordot(H, U, \
            axes = (2,0))*Pm).sum(axis=2).sum(axis=1).mean()
    
    energy_factor_p = (Pm*b_p).sum(axis=2).sum(axis=1).mean()
    
    
    energy = energy_factors_hh + energy_factors_h_mid + energy_factors_h_0 \
            + energy_factors_h_L + energy_factors_xh + energy_factors_x \
            + energy_factor_ph + energy_factor_p
    
    return energy

# get the energy with odd and even h marginalized separately
def get_odd_even_energy(X, P, H, W, V, U, b, b_0, b_L, d, Lambda, b_p, \
                        marginalize_visible):
    h_odd_marginalized = T.set_subtensor(H[:,1::2], \
                            update_odd_mu(X, P, H, W, V, U, b, b_L))
    h_even_marginalized = T.set_subtensor(H[:,::2], \
                            update_even_mu(X, P, H, W, V, U, b, b_0, b_L))
    
    if marginalize_visible:
        energy_h_odd_marginalized = get_energy(X, P, h_odd_marginalized, W, V, \
                            U, b, b_0, b_L, d, Lambda, b_p, \
                            x_marginalized = "even", \
                            p_marginalized = "even")
        energy_h_even_marginalized = get_energy(X, P, h_even_marginalized, W, \
                            V, U, b, b_0, b_L, d, Lambda, b_p, \
                            x_marginalized = "odd", \
                            p_marginalized = "odd")
    else:
        energy_h_odd_marginalized = get_energy(X, P, h_odd_marginalized, W, V, \
                            U, b, b_0, b_L, d, Lambda, b_p, \
                            x_marginalized = None, \
                            p_marginalized = None)
        energy_h_even_marginalized = get_energy(X, P, h_even_marginalized, W, \
                            V, U, b, b_0, b_L, d, Lambda, b_p, \
                            x_marginalized = None, \
                            p_marginalized = None)
    
    energy = 0.5*(energy_h_odd_marginalized + energy_h_even_marginalized)
    
    return energy



