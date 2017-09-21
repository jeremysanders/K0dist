#!/usr/bin/env python

from __future__ import division, print_function
from itertools import izip

import numpy as N

import h5py
import emcee
import scipy.optimize
import scipy.stats

_likes = _Kbins = _Kedges = None
_bestlike = -N.inf

allowneg = False
suffix = 'PosOnly'

def gaussLogLike(val, mu, sigma):
    """Log likelihood for a gaussian."""
    return (
        -0.5*N.log(2*N.pi)
          -N.log(sigma)
          -0.5*((val-mu)/sigma)**2
         )

def like(pars):
    """Calculate log likelihood for model."""
    global _likes, _Kbins, _Kedges, _bestlike

    if _likes is None:
        # Very ugly code to load the data into global variables, so
        # that we don't have to keep copying the data to the threads.
        filename = 'K0_dist_table_%s.hdf5' % suffix
        with h5py.File(filename, 'r') as f:
            _likes = N.array(f['likes'])
            _Kbins = N.array(f['Kbins'])
            _Kedges = N.array(f['Kedges'])

    mean1, mean2, sigma1, sigma2, skew1, skew2, bal = pars

    if sigma1 < 1 or sigma2 < 1 or sigma1 > 500 or sigma2 > 500:
        return -N.inf

    if mean1>_Kbins[-1] or mean2>_Kbins[-1]:
        return -N.inf

    minK = _Kbins[0] if allowneg else 0
    if mean1<=minK or mean2<=minK:
        return -N.inf

    # this Sigmoid function goes between 0 and 1
    sigm = 1/(1+N.exp(-bal))

    # this is the two-component pdf for the K0 values
    dist1 = scipy.stats.skewnorm(skew1, loc=mean1, scale=sigma1).pdf(_Kbins)
    dist2 = scipy.stats.skewnorm(skew2, loc=mean2, scale=sigma2).pdf(_Kbins)
    distprob = sigm*dist1 + (1-sigm)*dist2

    # renormalize PDF if we lose signal below 0 or above max
    distprob *= 1/N.trapz(distprob, x=_Kbins)

    # integrate to compute likelihood for each cluster
    likes = N.trapz(distprob*_likes, x=_Kbins, axis=1)

    # this is the total
    sumdistloglike = N.sum(N.log(likes))

    # priors for skew and the parameter
    skewprior = gaussLogLike(skew1,0,20) + gaussLogLike(skew2,0,20)
    balprior = gaussLogLike(bal,0,20)

    tot = sumdistloglike + skewprior + balprior

    if not N.isfinite(tot):
        tot = -N.inf

    if tot > _bestlike:
        _bestlike = tot
        print(_bestlike, pars)

    return tot

def runMCMC():
    """Run MCMC on the tabulated K0 posterior probability distributions."""

    # parameters are mean1, mean2, sigma1, sigma2, skew1, skew2, bal (sigmoid)
    p0 = N.array((0.1, 100, 20, 50, 0, 0, 0))

    # find best fitting parameters
    for i in xrange(5):
        for method in 'Nelder-Mead', 'Powell':
            retn = scipy.optimize.minimize(
                lambda p: -like(p),
                p0, method=method)
            p0 = retn.x

    print(p0)
    print('like', like(p0))

    ndim, nwalkers = len(p0), 200

    # setup initial parameters for each walker
    initpars = []
    while len(initpars) < nwalkers:
        p = N.random.normal(size=len(p0))*0.01 + p0
        if N.isfinite(like(p)):
            initpars.append(p)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, like, threads=8)

    print('Burning in')
    for i, result in enumerate(
        sampler.sample(initpars, iterations=2000, storechain=False)):
        print(i)
        pos = result[0]

    print('Sampling')
    sampler.reset()
    for i, result in enumerate(sampler.sample(pos, iterations=2000)):
        print(i)

    # write output chains to HDF5 file
    with h5py.File('K0_dist_chain_%s.hdf5' % suffix, 'w') as f:
        f.create_dataset(
            'chain', sampler.chain.shape,
            compression='gzip', compression_opts=9, shuffle=True)
        f.create_dataset(
            'like', sampler.lnprobability.shape,
            compression='gzip', compression_opts=9, shuffle=True)

        f['chain'][:,:,:] = sampler.chain
        f['like'][:,:] = sampler.lnprobability

def makeK0Dist():
    print('Computing distribution')
    # load chains from previously
    with h5py.File('K0_dist_chain_%s.hdf5' % suffix, 'r') as f:
        chain = N.array(f['chain'])

    # load 
    with h5py.File('K0_dist_table_%s.hdf5' % suffix, 'r') as f:
        Kbins = N.array(f['Kbins'])

    # sample random set of parameters from the chain
    chain = chain.reshape((-1, chain.shape[-1]))
    sample = N.random.randint(0, high=len(chain), size=40000)
    csample = chain[sample]

    out = []
    outcuml = []
    for pars in csample:
        mean1, mean2, sigma1, sigma2, skew1, skew2, bal = pars

        hprior1 = scipy.stats.skewnorm(skew1, loc=mean1, scale=sigma1)
        hprior2 = scipy.stats.skewnorm(skew2, loc=mean2, scale=sigma2)
        sigm = 1/(1+N.exp(-bal))
        distprob = sigm*hprior1.pdf(Kbins)+(1-sigm)*hprior2.pdf(Kbins)

        if not allowneg:
            distprob[Kbins<=0] = 0

        # renormalize PDF if we lose signal below 0 or above max
        distprob *= 1/N.trapz(distprob, x=_Kbins)

        out.append(distprob)

        cuml = N.cumsum(distprob * (_Kedges[1:]-_Kedges[:-1]))
        outcuml.append(cuml)

    # calculate percentiles and write to output data file
    med, lo, hi = N.percentile(out, [50, 15.85, 84.15], axis=0)
    prob = N.column_stack( (med, hi-med, lo-med) )
    med, lo, hi = N.percentile(outcuml, [50, 15.85, 84.15], axis=0)
    probcuml = N.column_stack( (med, hi-med, lo-med) )
    with h5py.File('K0_dist_meds_%s.hdf5' % suffix, 'w') as f:
        f['K0'] = Kbins
        f['prob'] = prob
        f['prob'].attrs['vsz_twod_as_oned'] = 1
        f['probcuml'] = probcuml
        f['probcuml'].attrs['vsz_twod_as_oned'] = 1

if __name__ == '__main__':
    runMCMC()
    makeK0Dist()
