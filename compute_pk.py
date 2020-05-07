from nbodykit.lab import *
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


# Reading configurations
config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = yaml.load(f)

input_path = load_config("input", "input_path")
displaced_catalog_fname = load_config("input", "displaced_catalog")
random_catalog_fname = load_config("input", "random_catalog")

output_path = load_config("output", "output_path")
output_name_pk = load_config("output", "output_name_pk")
output_name_pk_l = load_config("output", "output_name_pk_l")

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
    print("WARNING: The kmax value you specified ({} h/Mpc) exceeds the Nyquist wavenumber ({} h/Mpc).".format(kmax, k_Nyquist), file=sys.stderr)
dk = load_config("pk_config", "dk", 0.01)
Nmu = load_config("pk_config", "Nmu", 120)


t0 = time.time()

# Create catalog
names = ['x','y','z_red','w']
#names = ['x','y','z_red']

cat_d = CSVCatalog(input_path + displaced_catalog_fname, names)
cat_s = CSVCatalog(input_path + random_catalog_fname, names)
for cat in [cat_d, cat_s]:
    cat['RSDPosition'] = cat['x'][:,None] * [1, 0, 0] + cat['y'][:,None] * [0, 1, 0] + cat['z_red'][:,None] * [0, 0, 1]
    cat.attrs['BoxSize'] = BoxSize

# Create mesh (real space)
mesh_d = cat_d.to_mesh(window='tsc',Nmesh=Nmesh,compensated=True,interlaced=True,position='RSDPosition')
mesh_s = cat_s.to_mesh(window='tsc',Nmesh=Nmesh,compensated=True,interlaced=True,position='RSDPosition')

r_d = FFTPower(first=mesh_d,mode='2d',dk=dk,kmin=kmin,kmax=kmax,Nmu=Nmu,los=[0,0,1],poles=[0,2,4])
r_s = FFTPower(first=mesh_s,mode='2d',dk=dk,kmin=kmin,kmax=kmax,Nmu=Nmu,los=[0,0,1],poles=[0,2,4])
r_ds = FFTPower(first=mesh_d,second=mesh_s,mode='2d',
                dk=dk,kmin=kmin,kmax=kmax,Nmu=Nmu,los=[0,0,1],poles=[0,2,4])

Pkmu_d = r_d.power
Pkmu_s = r_s.power
Pkmu_ds = r_ds.power

poles_d = r_d.poles
poles_s = r_s.poles
poles_ds = r_ds.poles

# Writing output
f = open(output_path + output_name_pk,'w')
f.write('# Reconstruction power spectrum and multipoles\n')
f.write('# Displaced catalog: %s\n' % (displaced_catalog_fname))
f.write('# Random catalog: %s\n' % (random_catalog_fname))
f.write('# Estimated shot noise subtracted from power spectra\n')
f.write('# Estimated shot noise for displaced field: %.5f\n' % (Pkmu_d[:,0].attrs['shotnoise']))
f.write('# Estimated shot noise for random field: %.5f\n' % (Pkmu_s[:,0].attrs['shotnoise']))
f.write('# Code to generate this measurement in ' + __file__ + '\n')
f.write('# Boxsize = %.1f\n'  % BoxSize)
f.write('# Nmesh =  %i\n' % Nmesh)
f.write('# Binning = ' + binning + '\n')
f.write('# k mu pk-shotnoise Nmodes_d Nmodes_s Shotnoise_d Shotnoise_s\n')
for i in range(Pkmu_d.shape[1]):
    Pk_d = Pkmu_d[:,i]
    Pk_s = Pkmu_s[:,i]
    Pk_ds = Pkmu_ds[:,i]
    mu = Pkmu_d.coords['mu'][i]
    for j in range(len(Pk_d['k'])):
        k = Pk_d['k'][j]
        shotnoise_d = Pk_d.attrs['shotnoise']
        shotnoise_s = Pk_s.attrs['shotnoise']
        Pk_d_val = Pk_d['power'][j].real-shotnoise_d
        Pk_s_val = Pk_s['power'][j].real-shotnoise_s
        Pk_ds_val = Pk_ds['power'][j].real
        modes_d_val = Pkmu_d.data["modes"][:,i][j]
        modes_s_val = Pkmu_s.data["modes"][:,i][j]
        Pk_val = Pk_d_val + Pk_s_val - 2*Pk_ds_val

        f.write('%20.8e %20.8e %20.8e %i %i %20.8e %20.8e\n' % (k, mu, Pk_val, modes_d_val, modes_s_val, shotnoise_d, shotnoise_s))
f.close()

f = open(output_path + output_name_pk_l,'w')
f.write('# Reconstruction power spectrum and multipoles\n')
f.write('# Displaced catalog: %s\n' % (displaced_catalog_fname))
f.write('# Random catalog: %s\n' % (random_catalog_fname))
f.write('# Estimated shot noise subtracted from power spectra\n')
f.write('# Estimated shot noise for displaced field: %.5f\n' % (Pkmu_d[:,0].attrs['shotnoise']))
f.write('# Estimated shot noise for random field: %.5f\n' % (Pkmu_s[:,0].attrs['shotnoise']))
f.write('# Code to generate this measurement in ' + __file__ + '\n')
f.write('# Boxsize = %.1f\n'  % BoxSize)
f.write('# Nmesh =  %i\n' % Nmesh)
f.write('# Binning = ' + binning + '\n')
f.write('# k P0-shotnoise P2-shotnoise P4-shotnoise Nmodes_d Nmodes_s Shotnoise_d Shotnoise_s\n')
shotnoise_d = poles_d.attrs['shotnoise']
shotnoise_s = poles_s.attrs['shotnoise']
shotnoise = shotnoise_d + shotnoise_s
P0 = poles_d['power_0'].real + poles_s['power_0'].real - 2*poles_ds['power_0'].real - shotnoise
P2 = poles_d['power_2'].real + poles_s['power_2'].real - 2*poles_ds['power_2'].real - shotnoise
P4 = poles_d['power_4'].real + poles_s['power_4'].real - 2*poles_ds['power_4'].real - shotnoise
nmodes_d = poles_d.data['modes']
nmodes_s = poles_s.data['modes']

for i in range(len(P0)):
    f.write('%20.8e %20.8e %20.8e %20.8e %i %i %20.8e %20.8e\n' % (poles_d['k'][i], P0[i], P2[i], P4[i], nmodes_d[i], nmodes_s[i], shotnoise_d, shotnoise_s))

f.close()


print('Total time (s): ', time.time()-t0)

