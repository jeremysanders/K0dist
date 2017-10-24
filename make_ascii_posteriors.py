#!/usr/bin/env python

from __future__ import print_function
import h5py

# convert posteriors in HDF5 file into an ASCII format
# columns are cluster name, parameter, bin centre, bin inner edge,
#  bin outer edge, probability density and cumulative density

def main():
    inf = h5py.File('K0_dist_table_PosOnly.hdf5', 'r')
    outf = open('posteriors.txt', 'w')
    clusters = list(inf['cluster_names'])

    for par in ('K0', 'logK300', 'alphain', 'alphaout'):
        cent = inf['%s_bin_centres' % par]
        edges = inf['%s_bin_edges' % par]
        pdf = inf['%s_marg_pdf' % par]
        cpdf = inf['%s_cuml_prob' % par]

        for iclust, name in enumerate(clusters):
            for c, e1, e2, p, cp in zip(
                    cent, edges[:-1], edges[1:], pdf[iclust,:],
                    cpdf[iclust,:]):
                print(
                    '%s & %s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\' % (
                        name, par, c, e1, e2, p, cp
                        )
                )
                print(
                    '%16s %8s %6.4e %6.4e %6.4e %6.4e %6.4e' % (
                        name, par, c, e1, e2, p, cp
                        ),
                    file=outf
                    )

if __name__ == '__main__':
    main()
