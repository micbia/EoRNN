# coding: utf-8
import numpy as np, gzip, re, glob as gb, os, sys
sys.path.insert(0, '/its/home/mb756/python/tools21cm')

script, val = sys.argv
noc = int(val)

from tqdm import tqdm
from tools21cm import read_cbin, save_cbin, XfracFile, DensityFile, get_dens_redshifts, get_xfrac_redshifts
from tools21cm.irate_file import IonRateFile
from clump_functions import FindNearest, PrintTimeElapsed, SameValuesInArray, SaveMatrix, ReadMatrix, coarse_grid1

dens_path = '/research/prace/sph_smooth_cubepm_130627_12_6912_500Mpc_ext2/global/so/nc300/'
sourc_path = '/research/prace/subgrid_sources_500Mpc/300grids/'
cfact_path = '/research/prace/SubGridPhys/SimClumpMic_180614_500Mpc_nc300-MQL/scat/'
result_path = '/research/prace3/reion/500Mpc_RT/500Mpc_f5_8.2pS_300_stochastic_Cscat/results/'

redshit_in = get_dens_redshifts(dens_path)
redshift_out = get_xfrac_redshifts(result_path)
redshift_sourc = np.sort([float(os.path.split(f)[1].split('-')[0]) for f in gb.glob(sourc_path+'*wsubgrid_sources*')])

redshift = SameValuesInArray(SameValuesInArray(redshit_in, redshift_out), redshift_sourc)
np.savetxt('redshift.txt', redshift, fmt='%.3f', header='redshift in common between all data.')

sample_size = noc**3
dataset = np.zeros((sample_size*redshift.size, 7))
print('Creating EoRNN dataset from coarsed data with number of coarsening %d and for %d different redshift snapshot.' %(noc, redshift.size))
print('Dataset shape:', dataset.shape)

def ReadDataSources(fname):
    with gzip.open(fname, 'rb') as f:
        lines = f.readlines()
        data2txt = np.zeros(((len(lines)-1)/2, 7))
        for i in range(len(lines)/2):
            idx = 2*i+1
            text = re.sub(r"\s+", " ", lines[idx])+re.sub(r"\s+", " ", lines[idx+1])
            data2txt[i] = np.array(filter(None, text.split(' ')), dtype=float) 
    return data2txt


for i in tqdm(range(redshift.size)):
    z = redshift[i]
    red = z*np.ones(sample_size)
    
    # FEATURES
    nigm  = coarse_grid1(DensityFile(dens_path+'%.3fn_all.dat' %z).cgs_density, noc).flatten()
    cfact = coarse_grid1(np.power(10, read_cbin(cfact_path+'z%.3f_res1.667_scat.dat' %z)), noc).flatten()
    data = ReadDataSources(sourc_path+'%.3f-coarsest_wsubgrid_sources.dat.gz' %z)
    datasources9 = np.zeros((300,300,300))
    datasources89 = np.zeros((300,300,300))
    for val in data:
        a, b, c = int(val[0])-1, int(val[1])-1, int(val[2])-1
        datasources9[a, b, c] = val[3]
        datasources89[a, b, c] = val[4]
    
    nsourc9 = coarse_grid1(datasources9, noc).flatten()
    nsourc89 = coarse_grid1(datasources89, noc).flatten()
    
    # TARGET 
    irate = coarse_grid1(IonRateFile(result_path+'IonRates3_%.3f.bin' %z).irate, noc).flatten()
    xi = coarse_grid1(XfracFile(result_path+'xfrac3d_%.3f.bin' %z).xi, noc).flatten()
                        
    # FORMAT: z, nigm [cgs], cfact, source 9>, source 8:9, irate [1/s]
    dataset[i*sample_size:(i+1)*sample_size] = np.array([red, nigm, cfact, nsourc9, nsourc89, irate, xi]).T
    
    #dataset[:,i] = red
    #dataset[:,i+redshift.size] = nigm
    # ...
    
SaveMatrix('reionNNdataset_%dx%d_%d.dat' %(sample_size, redshift.size, dataset.shape[1]), dataset, size=dataset.shape)
