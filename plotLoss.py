import numpy as np, matplotlib.pyplot as plt, os 
from sys import argv

script, path = argv

os.chdir(path)

# Load Data
loss, loss_val = np.loadtxt('loss.txt'), np.loadtxt('val_loss.txt') 
r2, r2_val = np.loadtxt('r2.txt'), np.loadtxt('val_r2.txt') 
mae, mae_val = np.loadtxt('mae.txt'), np.loadtxt('val_mae.txt') 
lr = np.loadtxt('lr.txt') 

# Plot
fig1 = plt.figure(figsize=(800/50, 800/96), dpi = 96) 
fig1.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.25, right=0.93, left=0.08) 
  
ax1 = plt.subplot(1,2,1) 
ax1.set_ylabel('MSE'), ax1.set_xlabel('Epoch') 
ax1.plot(loss_val, color='cornflowerblue', label='Validation Loss')  
ax1.plot(loss, color='navy', label='Training Loss') 
#ax1.set_ylim(3e-3, 5e-4) 
ax1.set_xlim(-10, loss.size+10) 
ax3 = ax1.twinx() 
ax3.semilogy(lr, color='k', alpha=0.4, label='Learning Rate') 
ax3.set_ylabel('Learning Rate') 
lns, labs   = ax1.get_legend_handles_labels() 
lns2, labs2 = ax3.get_legend_handles_labels() 
ax1.legend(lns+lns2, labs+labs2, loc=1) 
  
ax2 = plt.subplot(1,2,2) 
ax2.set_ylabel(r'$R^2$'), ax2.set_xlabel('Epoch') 
ax2.plot(r2_val, color='lightgreen', label=r'Validation $R^2$') 
ax2.plot(r2, color='forestgreen', label=r'Training $R^2$') 
#ax2.set_ylim(0.84, 0.941) 
ax2.set_xlim(-10,loss.size+10)#, ax2.set_ylim(0, 1) 
  
ax4 = ax2.twinx() 
ax4.semilogy(lr, color='k', alpha=0.4, label='Learning Rate') 
ax4.set_ylabel('Learning Rate') 
lns, labs   = ax2.get_legend_handles_labels() 
lns2, labs2 = ax4.get_legend_handles_labels() 
ax2.legend(lns+lns2, labs+labs2, loc=3) 
plt.savefig('loss.png', bbox_inches='tight')
plt.show()
