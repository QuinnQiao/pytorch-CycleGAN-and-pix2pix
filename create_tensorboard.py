import os
import sys
import tensorboardX

assert len(sys.argv) == 3
lossfile = open(sys.argv[1])
# writer = tensorboardX.SummaryWriter(sys.argv[2])

num_per_epoch = 1231
w_cyc = 10.
w_idt = 10. * 0.5

for line in lossfile:
	if not line.startswith('(epoch:'):
		continue

	s = line.strip().split()
	epoch = int(s[1][:-1])
	iters = int(s[3][:-1])
	loss_dis_a = float(s[9])
	loss_gen_adv_a = float(s[11])
	loss_gen_cyc_a = float(s[13])
	loss_gen_idt_a = float(s[15])
	loss_dis_b = float(s[17])
	loss_gen_adv_b = float(s[19])
	loss_gen_cyc_b = float(s[21])
	loss_gen_idt_b = float(s[23])
	
	cur_iters = num_per_epoch * (epoch-1)
	cur_iters += iters

	print(epoch, cur_iters)


'''
(epoch: 1, iters: 100, time: 0.352, data: 0.184) D_A: 0.357 G_A: 0.364 cycle_A: 3.155 idt_A: 1.086 D_B: 0.381 G_B: 0.409 cycle_B: 2.254 idt_B: 1.358
        1         3          5            7           9          11             13           15         17         19             21           23
'''