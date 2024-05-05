import numpy as np
import torch
import torch.nn as nn
import logging
from utils import DataSaverHook, StopForwardException
from ptq4sam.quantization.quantized_module import QuantizedModule
from ptq4sam.quantization.fake_quant import LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase , AdaptiveGranularityQuantize
logger = logging.getLogger('ptq4sam')


def save_inp_oup_data(model, module, cali_data: list, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):

    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad():
        for i in range(len(cali_data)):
            # print(i,len(cali_data))
            try:
                _ = model.extract_feat(cali_data[i])
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0])
                else:
                    input_data = data_saver.input_store[0]
                    if isinstance(input_data,tuple):
                        if len(input_data) == 3:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu(),input_data[2].cpu()))
                        else:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu()))
                    else:
                        cached[0].append(input_data.cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    cached[1].append(data_saver.output_store.detach().cpu())
    # if store_inp:
    #     cached[0] = torch.cat([x for x in cached[0]])
    # if store_oup:
    #     cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache()
    return cached


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 module: QuantizedModule,
                 weight: float = 1.,
                 iters: int = 20000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.,
                 reg_weight=None,
                 reg_weight_lamb=0.1
                 ):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(iters, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.reg_weight=reg_weight
        self.reg_weight_lamb = reg_weight_lamb

    def __call__(self, pred, tgt):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
            w_reg_loss = 0
        else:
            round_loss = 0
            w_reg_loss = 0
            layer_len = 0
            for layer in self.module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if self.reg_weight is None:
                        round_vals = layer.weight_fake_quant.rectified_sigmoid()
                        round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        # total_loss = w_reg_loss
        if self.count % 500 == 0:
            logger.info('Total loss:\t{:.4f} (rec:{:.4f}, round:{:.4f}, rw:{:.4f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), float(w_reg_loss), b, self.count))
        return total_loss


def lp_loss(pred, tgt, p=2.0):
    """
    loss function
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()


def reconstruction(model, fp_model, module, fp_module, cali_data, config):
    device = next(module.parameters()).device
    # get data first
    quant_inp, _ = save_inp_oup_data(model, module, cali_data, store_inp=True, store_oup=False, bs=config.batch_size, keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data(fp_model, fp_module, cali_data, store_inp=True, store_oup=True, bs=config.batch_size, keep_gpu=config.keep_gpu)
    # prepare for up or down tuning
    w_para, a_para = [], []
    
    # # for the bimodal block, add the gamma parameter
    # gamma_para = []
    # if hasattr(module,'gamma') and config.gamma_tune:
    #     gamma_para.append(module.gamma)

    for name, layer in module.named_modules():
        only4flag = ('only4' not in config.keys()) or (not config.only4) or (config.only4 and ('k_proj' in name or 'q_proj' in name))
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
            layer.drop_prob = config.drop_prob
            if only4flag:
                if isinstance(layer, LSQFakeQuantize):
                    a_para += [layer.scale]
                if isinstance(layer, LSQPlusFakeQuantize):
                    a_para += [layer.scale]
                    a_para += [layer.zero_point]
                if isinstance(layer, AdaptiveGranularityQuantize):
                    a_para += [layer.scale]
    
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        # a_opt = torch.optim.Adam([{"params":a_para,"lr":config.scale_lr},{"params":module.gamma,"lr":config.scale_lr}])
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    else:
        w_opt = None

    # if len(gamma_para) != 0:
    #     gamma_opt = torch.optim.Adam(gamma_para, lr=config.gamma_lr)
    # else:
    #     gamma_opt = None
    
    logger.info(name)
    logger.info(type(module))
    logger.info(len(a_para))


    if len(a_para) == 0 and len(w_para) == 0:
        logger.info('skip opt')
        del fp_inp,fp_oup,quant_inp
        torch.cuda.empty_cache()
        for name, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if weight_quantizer.adaround:
                    weight_quantizer = layer.weight_fake_quant
                    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                    weight_quantizer.adaround = False
            if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
                layer.drop_prob = 1.0
        return

    loss_func = LossFunction(module=module, weight=config.weight, iters=config.iters, b_range=config.b_range,
                             warm_up=config.warm_up)

    from mmdet.utils import build_ddp,build_dp
    import os
    module_ddp = build_dp(module, 'cuda', device_ids=[0])
    # module_ddp = build_ddp(
    #     module,
    #     'cuda',
    #     device_ids=[int(os.environ['LOCAL_RANK'])],
    #     broadcast_buffers=False)  
    try:
        for i in range(len(fp_oup)):
            fp_oup[i] = fp_oup[i].cuda()
        
        for i,t in enumerate(fp_inp):
            if isinstance(t,tuple):
                if len(t) == 3:
                    fp_inp[i] = (t[0].cuda(),t[1].cuda(),t[2].cuda())
                else:
                    fp_inp[i] = (t[0].cuda(),t[1].cuda())
            else:
                fp_inp[i] = t.cuda()
        
        for i,t in enumerate(quant_inp):
            if isinstance(t,tuple):
                if len(t) == 3:
                    quant_inp[i] = (t[0].cuda(),t[1].cuda(),t[2].cuda())
                else:
                    quant_inp[i] = (t[0].cuda(),t[1].cuda())
            else:
                quant_inp[i] = t.cuda()
    except:
        in_cpu = 32
        logger.info('in_cpu 32')
        for i in range(len(fp_oup)):
            fp_oup[i] = fp_oup[i].cuda()
        
        for i,t in enumerate(fp_inp):
            if i < in_cpu:
                if isinstance(t,tuple):
                    if len(t) == 3:
                        fp_inp[i] = (t[0].cpu(),t[1].cpu(),t[2].cpu())
                    else:
                        fp_inp[i] = (t[0].cpu(),t[1].cpu())
                else:
                    fp_inp[i] = t.cpu()
            else:
                if isinstance(t,tuple):
                    if len(t) == 3:
                        fp_inp[i] = (t[0].cuda(),t[1].cuda(),t[2].cuda())
                    else:
                        fp_inp[i] = (t[0].cuda(),t[1].cuda())
                else:
                    fp_inp[i] = t.cuda()
        
        for i,t in enumerate(quant_inp):
            if i < in_cpu:
                if isinstance(t,tuple):
                    if len(t) == 3:
                        quant_inp[i] = (t[0].cpu(),t[1].cpu(),t[2].cpu())
                    else:
                        quant_inp[i] = (t[0].cpu(),t[1].cpu())
                else:
                    quant_inp[i] = t.cpu()    
            else:
                if isinstance(t,tuple):
                    if len(t) == 3:
                        quant_inp[i] = (t[0].cuda(),t[1].cuda(),t[2].cuda())
                    else:
                        quant_inp[i] = (t[0].cuda(),t[1].cuda())
                else:
                    quant_inp[i] = t.cuda()

        
    sz = len(cali_data)
    for i in range(config.iters):
        idx = torch.randint(0, sz, (1, ))
        if config.drop_prob < 1.0:
            # cur_quant_inp = quant_inp[idx].to(device)
            # cur_quant_inp = quant_inp[idx]
            cur_quant_inp = quant_inp[idx]
            cur_fp_inp = fp_inp[idx]
    
            # cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
            if isinstance(cur_quant_inp, torch.Tensor):
                cur_quant_inp = cur_quant_inp.cuda()
                cur_fp_inp = cur_fp_inp.cuda()
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp.cuda())
            elif len(cur_quant_inp) == 2:
                
                cur_quant_inp = (cur_quant_inp[0].cuda(), cur_quant_inp[1].cuda())
                cur_fp_inp = (cur_fp_inp[0].cuda(), cur_fp_inp[1].cuda())
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp = (cur_inp0,cur_inp1)
            else:
                cur_quant_inp = (cur_quant_inp[0].cuda(), cur_quant_inp[1].cuda(), cur_quant_inp[2].cuda())
                cur_fp_inp = (cur_fp_inp[0].cuda(), cur_fp_inp[1].cuda(), cur_fp_inp[2].cuda())
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp2 = torch.where(torch.rand_like(cur_quant_inp[2]) < config.drop_prob, cur_quant_inp[2], cur_fp_inp[2])
                cur_inp = (cur_inp0,cur_inp1,cur_inp2)
        else:
            cur_inp = quant_inp[idx]
        cur_fp_oup = fp_oup[idx]
        if a_opt:
            a_opt.zero_grad()
        # if gamma_opt:
        #     gamma_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()
        # import pdb;pdb.set_trace()
        cur_quant_oup = module_ddp(cur_inp)
        err = loss_func(cur_quant_oup, cur_fp_oup)
        cur_inp = None
        cur_quant_oup = None
        torch.cuda.empty_cache()
        err.backward() # del cur_inp cur_quant_oup
        if w_opt:
            w_opt.step()
        # if gamma_opt:
        #     gamma_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()
    
    del fp_inp,fp_oup,quant_inp,cur_fp_oup
    torch.cuda.empty_cache()
    
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if weight_quantizer.adaround:
                weight_quantizer = layer.weight_fake_quant
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0
