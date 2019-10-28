import random, numpy as np, pandas as pd, glob as gb, os, sys, gzip, re, matplotlib.pyplot as plt

from tqdm import tqdm
sys.path.insert(0, '/home/michele/python/tools21cm')
from tools21cm import read_cbin, XfracFile, DensityFile, VelocityFile, get_dens_redshifts, get_xfrac_redshifts
from tools21cm.irate_file import IonRateFile
from clump_functions import FindNearest, PrintTimeElapsed, SameValuesInArray, SaveMatrix, ReadMatrix, SaveBinaryFile, OpenBinaryFile

from keras.models import load_model
from keras import backend as K
from sklearn import preprocessing


script, redshift, path = sys.argv
z = float(redshift)

rho_c0 = 9.204552e-30

def R2(y_true, y_pred):
    ''' R^2 (coefficient of determination) regression score function. To avoid NaN, added 1e-8 to denominator '''
    SS_num =  K.sum(K.square(y_true-y_pred))
    SS_den = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_num/(SS_den + K.epsilon())

dens_path = './data/inputs/'
sourc_path = dens_path
cfact_path = dens_path
result_path = dens_path
#dens_path = '/research/prace/sph_smooth_cubepm_130627_12_6912_500Mpc_ext2/global/so/nc300/'
#sourc_path = '/research/prace/subgrid_sources_500Mpc/300grids/'
#cfact_path = '/research/prace/SubGridPhys/SimClumpMic_180614_500Mpc_nc300-MQL/scat/'
#result_path = '/research/prace3/reion/500Mpc_RT/500Mpc_f5_8.2pS_300_stochastic_Cscat/results/'

nigm  = np.log10(DensityFile(dens_path+'%.3fn_all.dat' %z).cgs_density.flatten()/rho_c0)
cfact = read_cbin(cfact_path+'z%.3f_res1.667_scat.dat' %z).flatten()
nsourc9 = OpenBinaryFile(sourc_path+'%.3f_1.667_m9.dat' %z)
#nsourc89 = OpenBinaryFile(sourc_path+'%.3f_1.667_m89.dat' %z)
irate = np.log10((IonRateFile(result_path+'IonRates3_%.3f.bin' %z).irate).flatten())

# load model
model = load_model(path+'model/model.h5', custom_objects={'R2': R2})
model.summary()
print('Metrics of loaded model:\n', model.metrics_names)

expt_out = np.loadtxt(path+'expected_outputs.txt')
scale_y = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(expt_out.reshape(-1,1))
predictions = model.predict(np.array([nigm, cfact, nsourc9]).T, verbose=1)
predictions = scale_y.inverse_transform(predictions)
SaveBinaryFile('%.3firate_prediction.dat' %z, [300, 300,300], np.array(predictions).reshape((300, 300, 300)))
print('Prediction:\n', 'min:\t', np.min(predictions), 'max:\t', np.max(predictions))

#score = model.evaluate(np.array([nigm, cfact, nsourc9]).T, irate)
#print(score)

plt.figure(figsize=(10,8))
plt.plot(irate, np.log10(predictions), 'x', zorder=1)
plt.plot(np.linspace(irate.min(),irate.max(),3), np.linspace(irate.min(),irate.max(),3), 'k--')
plt.grid()
plt.savefig('%.3f_irate_predictions.png' %z, bbox_inches='tight')
#plt.show()
