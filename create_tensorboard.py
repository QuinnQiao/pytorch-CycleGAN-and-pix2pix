import os
import sys
import tensorboardX

assert len(sys.argv) == 3
lossfile = open(sys.argv[1])
writer = tensorboardX.SummaryWriter(sys.argv[2])

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

	writer.add_scalar('loss_dis_a', loss_dis_a, cur_iters)
	writer.add_scalar('loss_dis_b', loss_dis_b, cur_iters)
	tot_d = loss_dis_a + loss_dis_b
	writer.add_scalar('loss_dis_tot', tot_d, cur_iters)

	writer.add_scalar('loss_gen_adv_a', loss_gen_adv_a, cur_iters)
	writer.add_scalar('loss_gen_adv_b', loss_gen_adv_b, cur_iters)
	writer.add_scalar('loss_gen_cyc_a', loss_gen_cyc_a / w_cyc, cur_iters)
	writer.add_scalar('loss_gen_cyc_b', loss_gen_cyc_b / w_cyc, cur_iters)
	writer.add_scalar('loss_gen_idt_a', loss_gen_idt_a / w_idt, cur_iters)
	writer.add_scalar('loss_gen_idt_b', loss_gen_idt_b / w_idt, cur_iters)
	tot_g = loss_gen_adv_a + loss_gen_adv_b + loss_gen_cyc_a + loss_gen_cyc_b + loss_gen_idt_a + loss_gen_idt_b
	writer.add_scalar('loss_gen_tot', tot_g, cur_iters)


'''
(epoch: 1, iters: 100, time: 0.352, data: 0.184) D_A: 0.357 G_A: 0.364 cycle_A: 3.155 idt_A: 1.086 D_B: 0.381 G_B: 0.409 cycle_B: 2.254 idt_B: 1.358
        1         3          5            7           9          11             13           15         17         19             21           23
'''