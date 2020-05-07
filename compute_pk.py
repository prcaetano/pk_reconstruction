import os
from nbodykit.lab import CSVCatalog
from fftpower_reconstruction import FFTPowerReconstruction
import numpy as np
import time
import sys
import yaml


def load_config(section, name, default=None):
    try:
        return config[section][name]
    except KeyError:
        if default is not None:
            return default
        else:
            error_msg = "Required configuration {}:{} not fount, and no default available".format(section, name)
            raise RuntimeError(error_msg)


if __name__ == "__main__":
    # Reading configurations
    try:
        config_path = sys.argv[1]
    except IndexError:
        print("Usage: compute_pk <configuration YAML file>")
        exit(1)
    with open(config_path, "r") as f:
        config = yaml.load(f)

    path = load_config("output", "output_path")
    displaced_catalog_fname = load_config("output", "displaced_catalog_fname")
    random_catalog_fname = load_config("output", "shifted_random_catalog_fname")
    has_weight_col = load_config("output", "has_weight_col", True)

    output_name_pk = load_config("output", "output_name_pk")
    output_name_pk_l = load_config("output", "output_name_pk_multipoles")

    Nmesh = load_config("pk_config", "Nmesh")
    BoxSize = load_config("simulation_config", "BoxSize")

    # Need mesh to be big enough so that nyquist frequency is > kmax
    # pi * Nmesh/Lbox = 3.22 h Mpc^{-1} for Nmesh = 1024 and Lbox = 1000 h^{-1} Mpc, 
    # vs kmax = 2.262 h Mpc^{-1}
    # Interlacing + TSC gives very unbiased results up to the Nyquist frequency:
    # https://nbodykit.readthedocs.io/en/latest/cookbook/interlacing.html
    k_Nyquist = np.pi * Nmesh/BoxSize

    binning = load_config("pk_config", "binning", "linear")
    kmin = load_config("pk_config", "kmin", 0.0)
    kmax = load_config("pk_config", "kmax", k_Nyquist)
    if kmax > k_Nyquist:
        print("WARNING: The kmax value you specified ({} h/Mpc) exceeds "
              " the Nyquist wavenumber ({} h/Mpc).".format(kmax, k_Nyquist),
              file=sys.stderr)
    dk = load_config("pk_config", "dk", 0.01)
    Nmu = load_config("pk_config", "Nmu", 120)

    # Begin timing
    t0 = time.time()

    # Create catalog
    if has_weight_col:
        names = ['x','y','z','w']
    else:
        names = ['x','y','z']
    cat_d = CSVCatalog(os.path.join(path, displaced_catalog_fname), names)
    cat_s = CSVCatalog(os.path.join(path, random_catalog_fname), names)
    for cat in [cat_d, cat_s]:
        cat['RSDPosition'] = cat['x'][:,None]*[1,0,0] + cat['y'][:,None]*[0,1,0] + cat['z'][:,None]*[0,0,1]
        cat.attrs['BoxSize'] = BoxSize

    # Create mesh (real space)
    mesh_d = cat_d.to_mesh(window='tsc',Nmesh=Nmesh,compensated=True,interlaced=True,position='RSDPosition')
    mesh_s = cat_s.to_mesh(window='tsc',Nmesh=Nmesh,compensated=True,interlaced=True,position='RSDPosition')

    r = FFTPowerReconstruction(displaced=mesh_d,random=mesh_s,mode='2d',dk=dk,kmin=kmin,kmax=kmax,Nmu=Nmu,
                               los=[0,0,1],poles=[0,2,4])
    Pkmu = r.power
    poles = r.poles

    # Writing output pk
    with open(os.path.join(path, output_name_pk), "w") as f:
        f.write('# Reconstruction power spectrum and multipoles\n')
        f.write('# Displaced catalog: %s\n' % (displaced_catalog_fname))
        f.write('# Random catalog: %s\n' % (random_catalog_fname))
        f.write('# Estimated shot noise subtracted from power spectra\n')
        f.write('# Estimated shot noise for reconstructed field: %.5f\n' % (Pkmu[:,0].attrs['shotnoise']))
        f.write('# Code to generate this measurement in ' + __file__ + '\n')
        f.write('# Boxsize = %.1f\n'  % BoxSize)
        f.write('# Nmesh =  %i\n' % Nmesh)
        f.write('# Binning = ' + binning + '\n')
        f.write('# k mu pk-shotnoise shotnoise\n')
        for i in range(Pkmu.shape[1]):
            Pk = Pkmu[:,i]
            mu = Pkmu.coords['mu'][i]
            for j in range(len(Pk['k'])):
                k = Pk['k'][j]
                shotnoise = Pk.attrs['shotnoise']
                Pk_val = Pk['power'][j].real - shotnoise
                f.write('%20.8e %20.8e %20.8e %20.8e\n' % (k, mu, Pk_val, shotnoise))

    # Writing output multipoles
    with open(os.path.join(path, output_name_pk_l), "w") as f:
        f.write('# Reconstruction power spectrum and multipoles\n')
        f.write('# Displaced catalog: %s\n' % (displaced_catalog_fname))
        f.write('# Random catalog: %s\n' % (random_catalog_fname))
        f.write('# Estimated shot noise subtracted from power spectra\n')
        f.write('# Estimated shot noise for reconstructed field: %.5f\n' % (Pkmu[:,0].attrs['shotnoise']))
        f.write('# Code to generate this measurement in ' + __file__ + '\n')
        f.write('# Boxsize = %.1f\n'  % BoxSize)
        f.write('# Nmesh =  %i\n' % Nmesh)
        f.write('# Binning = ' + binning + '\n')
        f.write('# k P0-shotnoise P2-shotnoise P4-shotnoise shotnoise\n')
        shotnoise = poles.attrs['shotnoise']
        P0 = poles['power_0'].real - shotnoise
        P2 = poles['power_2'].real - shotnoise
        P4 = poles['power_4'].real - shotnoise
        for i in range(len(P0)):
            f.write('%20.8e %20.8e %20.8e %20.8e %20.8e\n' % (poles['k'][i], P0[i], P2[i], P4[i], shotnoise))

    print('Total time (s): ', time.time()-t0)

