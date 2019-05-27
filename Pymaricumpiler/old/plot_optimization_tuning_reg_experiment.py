import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

path_no_reg = '/Users/henrypinkard/Desktop/optimize_logs/'
path_reg = '/Users/henrypinkard/Desktop/optimize_tune_reg/'
data = {}

for filename in os.listdir(path_reg) + os.listdir(path_no_reg):
    if filename.startswith('.') or filename.startswith('frame_7') or filename.endswith('prefilter_True.txt'):
        continue
    if filename.startswith('frame_0__lr_0.5') or filename.startswith('frame_0__lr_5') or filename.startswith('frame_0__lr_1'):
        continue
    if filename.startswith('frame_0'):
        data[filename] = np.genfromtxt(path_no_reg + filename, delimiter=',')
    else:
        data[filename] = np.genfromtxt(path_reg + filename, delimiter=',')

lrs = set()
stack_regs = set()
stitch_regs = set()

lines = {}

def plotit(data, title, index):
    """
    Plot one type of metric for all conditions
    """
    #0 stitch loss    1 stack loss    2 stich rms    3 stack rms
    legend = []
    for key in data.keys():
        # if ('0.1' not in key) and ('0.01' not in key):
        #     continue

        if key.startswith('frame_0'):
            lr = key.split('_')[4]
            stack_reg = '0'
            stitch_reg = '0'
        elif key.startswith('Adam'):
            lr = key[:-4]
            stitch_reg = '0.01'
            stack_reg = '0.01'
        elif key.startswith('different'):
            lr = '0'
            stitch_reg = '0.01'
            stack_reg = '0.01'
        else:
            lr, stitch_reg, stack_reg = [key.replace('.txt', '').split('_')[i] for i in [1, 5, 9]]
        # if not (lr == '1' or lr == '3' or lr == '9' or lr == '10' or lr == '0.1' or lr == '0.01' or lr == '0'):
        #     continue
        if not (stitch_reg == '0' or stitch_reg == '0.01' ):
            continue
        if not (stack_reg == '0' or stack_reg == '0.01' ):
            continue


        lrs.add(lr)
        stack_regs.add(stack_reg)
        stitch_regs.add(stitch_reg)

        # if lr == '0.01':
        #     linestyle = ':'
        # elif lr == '0.1':
        #     linestyle = '--'
        # elif lr == '10':
        #     linestyle = '-.'
        # else:
        #     linestyle = '-'

        if stitch_reg == '0':
            color = 'k'
        elif stitch_reg == '0.01':
            color = 'b'
        elif stitch_reg == '0.1':
            color = 'g'
        elif stitch_reg == '1':
            color = 'r'
        # elif stitch_reg == '0.01':
        #     color = 'm'
        # elif stitch_reg == '0.0001':
        #     color = 'y'


        if stack_reg == '0':
            linestyle = ':'
        elif stack_reg == '0.01':
            linestyle = '--'
        elif stack_reg == '1':
            linestyle = '-.'
        else:
            linestyle = '-'

        # if stitch_reg == None:
        key2 = (lr, stack_reg, stitch_reg)
        if key2 not in lines:
            lines[key2] = []


        line = plt.plot(data[key][:, index], linestyle + color)
        lines[key2].append(line[0])

        legend.append('LR: ' + lr + ' Stitch: ' + str(stitch_reg) + '  Stack: ' + str(stack_reg))
    plt.title(title)
    plt.legend(legend)


plt.figure()
plt.subplot(221)
plotit(data, 'stitch loss', 0)
plt.subplot(222)
plotit(data, 'stack loss', 1)
plt.subplot(223)
plotit(data, 'stitch rms', 2)
plt.subplot(224)
plotit(data, 'stack rms', 3)


lrs = list(lrs)
stack_regs = list(stack_regs)
stitch_regs = list(stitch_regs)


rax = plt.axes([0, 0.1, 0.1, 0.35])
check_lr = CheckButtons(rax, lrs, len(lrs) * [True])

rax = plt.axes([0, 0.5, 0.05, 0.15])
check_stack_reg = CheckButtons(rax, stack_regs, len(stack_regs) * [True])

rax = plt.axes([0, 0.7, 0.05, 0.15])
check_stitch_reg = CheckButtons(rax, stitch_regs, len(stitch_regs) * [True])

def update_lines(discar_param):
    for key in lines.keys():
        lr_vis = check_lr.get_status()[lrs.index(key[0])]
        stack_reg_vis = check_stack_reg.get_status()[stack_regs.index(key[1])]
        stitch_reg_vis = check_stitch_reg.get_status()[stitch_regs.index(key[2])]
        for line in lines[key]:
            line.set_visible(lr_vis and stack_reg_vis and stitch_reg_vis)
    plt.draw()

check_lr.on_clicked(update_lines)
check_stack_reg.on_clicked(update_lines)
check_stitch_reg.on_clicked(update_lines)


print(lrs)
print(stack_regs)
print(stitch_regs)

plt.show()
pass

log = data['Adam0.1.txt']
min_loss0 = log[0, 0]
min_loss1 = log[0, 1]
new_min_iter = 0
for i in np.arange(log.shape[0]):
    if min_loss0 > log[i, 0]:
        min_loss0 = log[i, 0]
        new_min_iter = 0
    if min_loss1 > log[i, 1]:
        min_loss1 = log[i, 1]
        new_min_iter = 0
    new_min_iter = new_min_iter + 1
    if new_min_iter == 10:
        break
print(i, min_loss0, min_loss1)


