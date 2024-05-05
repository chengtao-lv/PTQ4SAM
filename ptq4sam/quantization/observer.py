import numpy as np     # noqa: F401
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from .util_quant import fake_quantize_per_tensor_affine, fake_quantize_per_channel_affine, fake_logquantize_per_tensor_affine


def _transform_to_ch_axis(x, ch_axis):
    if ch_axis == -1:
        return x
    else:
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[ch_axis] = 0
        new_axis_list[0] = ch_axis
        x_channel = x.permute(new_axis_list)
        y = torch.flatten(x_channel, start_dim=1)
        return y


class ObserverBase(nn.Module):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(ObserverBase, self).__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def set_bit(self, bit):
        self.bit = bit
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int, device=device)
        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point


class MinMaxObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset.
    '''

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)


class MinMaxObserver2(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset.
    '''

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MinMaxObserver2, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        self.min_val = min_val_cur
        self.max_val = max_val_cur
        return min_val_cur, max_val_cur


class AvgMinMaxObserver(ObserverBase):
    """
    Average min/max calibration
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(AvgMinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0 
        assert self.ch_axis == -1

    def forward(self, x_orig):
        r"""Records the average minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.cnt + min_val_cur
            self.max_val = self.max_val * self.cnt + max_val_cur
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt

class AvgSignMinMaxObserver(ObserverBase):
    """
    Average min/max calibration
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(AvgMinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0 
        assert self.ch_axis == -1

    def forward(self, x_orig):
        r"""Records the average minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = torch.min(self.min_val, min_val_cur)
            self.max_val = self.max_val * self.cnt + max_val_cur
        self.cnt += 1
        # self.min_val /= self.cnt
        self.max_val /= self.cnt

class MSEObserver(ObserverBase):
    # grid search: more accurate but slow
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MSEObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = 2.4
        self.num = 100  # candidate num
        self.one_side_dist = None  # 'pos', 'neg', 'no'

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if self.ch_axis == -1:
            return x.mean()
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            return y.mean(1)

    def loss_fx(self, x, new_min, new_max):
        # should also consider channel here
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        if self.ch_axis != -1:
            x_q = fake_quantize_per_channel_affine(
                x, scale.data, zero_point.data.int(), self.ch_axis,
                self.quant_min, self.quant_max)
        else:
            x_q = fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()),
                self.quant_min, self.quant_max)
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def perform_2D_search(self, x):
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / float(self.quant_max - self.quant_min)
            # enumerate zp
            for zp in range(self.quant_min, self.quant_max + 1):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                score = self.loss_fx(x, new_min, new_max)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            score = self.loss_fx(x, new_min, new_max)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'

        if self.one_side_dist != 'no' or self.symmetric:  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.perform_2D_search(x)
        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)


class AvgMSEObserver(MSEObserver):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(AvgMSEObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0
        assert self.ch_axis == -1

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'

        if self.one_side_dist != 'no' or self.symmetric:  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.perform_2D_search(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.cnt + best_min
            self.max_val = self.max_val * self.cnt + best_max
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class MSEFastObserver(ObserverBase):
    # golden section search here
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MSEFastObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = 2.4
        self.num = 100  # candidate num
        self.one_side_dist = None  # 'pos', 'neg', 'no'

    def lp_loss(self, pred, tgt, p=2.0):
        return (pred - tgt).abs().pow(p).mean()

    def loss_fx(self, x, new_min, new_max):
        # only consider tensor here
        new_min = torch.tensor(new_min).cuda()
        new_max = torch.tensor(new_max).cuda()
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        x_q = fake_quantize_per_tensor_affine(
                    x, scale.item(), int(zero_point.item()),
                    self.quant_min, self.quant_max)
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def golden_asym_shift_loss(self, shift, xrange, x, x_min, x_max):
        tmp_min = 0.0
        tmp_max = xrange
        new_min = tmp_min - shift
        new_max = tmp_max - shift
        return self.loss_fx(x, new_min, new_max).cpu().numpy()

    def golden_asym_range_loss(self, xrange, x, x_min, x_max):
        tmp_delta = xrange / float(self.quant_max - self.quant_min)
        max_shift = tmp_delta * self.quant_max
        min_shift = tmp_delta * self.quant_min
        result = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(xrange, x, x_min, x_max),
            bounds=(min_shift, max_shift),
            method='Bounded',
        )
        return result.fun

    def golden_sym_range_loss(self, xrange, x):
        new_min = 0.0 if self.one_side_dist == 'pos' else -xrange
        new_max = 0.0 if self.one_side_dist == 'neg' else xrange
        return self.loss_fx(x, new_min, new_max).cpu().numpy()

    def golden_section_search_2D_channel(self, x, x_min, x_max):
        xrange = x_max - x_min
        result = minimize_scalar(
            self.golden_asym_range_loss,
            args=(x, x_min, x_max),
            bounds=(min(0.1, 0.01 * xrange.item()), xrange.item()),
            method='Bounded',
        )
        final_range = result.x
        tmp_min = 0.0
        tmp_max = final_range
        tmp_delta = final_range / float(self.quant_max - self.quant_min)
        max_shift = tmp_delta * self.quant_max
        min_shift = tmp_delta * self.quant_min
        subresult = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(final_range, x, x_min, x_max),
            bounds=(min_shift, max_shift),
            method='Bounded',
        )
        final_shift = subresult.x
        best_min = max(tmp_min - final_shift, x_min)
        best_max = min(tmp_max - final_shift, x_max)
        return torch.tensor(best_min).cuda(), torch.tensor(best_max).cuda()

    def golden_section_search_1D_channel(self, x, x_min, x_max):
        xrange = torch.max(x_min.abs(), x_max)
        result = minimize_scalar(
            self.golden_sym_range_loss,
            args=(x, ),
            bounds=(min(0.1, 0.01 * xrange.item()), xrange.item()),
            method='Bounded',
        )
        final_range = result.x
        best_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -torch.tensor(final_range)
        best_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else torch.tensor(final_range)
        return torch.tensor(best_min).cuda(), torch.tensor(best_max).cuda()

    def golden_section_2D_search(self, x):
        if self.ch_axis == -1:
            x_min, x_max = torch._aminmax(x)
            x_min, x_max = self.golden_section_search_2D_channel(x, x_min, x_max)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            for ch, val in enumerate(y):
                x_min[ch], x_max[ch] = self.golden_section_search_2D_channel(
                    y[ch], x_min[ch], x_max[ch])
        return x_min, x_max

    def golden_section_1D_search(self, x):
        if self.ch_axis == -1:
            x_min, x_max = torch._aminmax(x)
            x_min, x_max = self.golden_section_search_1D_channel(x, x_min, x_max)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            for ch, val in enumerate(y):
                x_min[ch], x_max[ch] = self.golden_section_search_1D_channel(
                    y[ch], x_min[ch], x_max[ch])
        return x_min, x_max

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'

        if self.one_side_dist != 'no' or self.symmetric:  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.golden_section_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.golden_section_2D_search(x)
        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)


class AvgMSEFastObserver(MSEFastObserver):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0
        assert self.ch_axis == -1

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'

        if self.one_side_dist != 'no' or self.symmetric:  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.golden_section_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.golden_section_2D_search(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.cnt + best_min
            self.max_val = self.max_val * self.cnt + best_max
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt

class LogAvgMSEFastObserver(AvgMSEFastObserver):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(bit, symmetric, ch_axis)
        self.register_buffer("clip_rate", torch.tensor(float("inf")))
        self.register_buffer("best_tau_scales", torch.tensor(float("inf")))
        self.register_buffer("tau_errors", torch.tensor(float("inf")))
        self.taus = [2**i for i in range(3)]
        self.value = None
        

    def loss_fx(self, x, new_min, new_max, alpha):
        scale = torch.tensor(new_max).cuda()
        x_q = fake_logquantize_per_tensor_affine(
                    x, scale.item(), self.quant_min, self.quant_max, alpha)
        # if self.v0 is None:
        #     score = self.lp_loss(x_q, x, p=self.p)
        # else:
        #     import warnings
        #     warnings.warn('@v0')
        #     # print('@v0')
        score = self.lp_loss(x_q @ self.value, x @ self.value, p=self.p)
        return score
    
    def forward(self, x_orig, value=None):
        if value is not None:
            self.value = value
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'

        if self.one_side_dist != 'no' or self.symmetric:  # one-side distribution or symmetric value for 1-d search
            best_tau_scales, tau_errors = self.golden_section_1D_search(x)
        
        if self.best_tau_scales.numel() <= 1 and self.best_tau_scales.isinf():
            self.best_tau_scales = best_tau_scales
            self.tau_errors = tau_errors
        else:
            self.best_tau_scales = self.best_tau_scales * self.cnt + best_tau_scales
            self.tau_errors = self.tau_errors + tau_errors

        self.cnt += 1
        self.best_tau_scales /= self.cnt

    def golden_section_1D_search(self, x):
        if self.ch_axis == -1:
            x_min, x_max = torch._aminmax(x)
            x_max, errors = self.golden_section_search_1D_channel(x, x_min, x_max)
        return x_max, errors

    def golden_section_search_1D_channel(self, x, x_min, x_max):
        xrange = torch.max(x_min.abs(), x_max)
        
        best_tau_scales = []
        tau_errors = []
        
        for tau in self.taus:
            result = minimize_scalar(
                self.golden_sym_range_loss,
                args=(x, tau),
                bounds=(min(0.1, 0.01 * xrange.item()), xrange.item()),
                method='Bounded',
            )
            final_range = result.x
            # best_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -torch.tensor(final_range)
            best_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else torch.tensor(final_range)
            # best_min_alphas.append(torch.tensor(best_min).cuda())
            best_tau_scales.append(torch.tensor(best_max))
            tau_errors.append(torch.tensor(result.fun))
        
        return torch.tensor(best_tau_scales).cuda(), torch.tensor(tau_errors).cuda()


    def golden_sym_range_loss(self, xrange, x, tau):
        new_min = 0.0 if self.one_side_dist == 'pos' else -xrange
        new_max = 0.0 if self.one_side_dist == 'neg' else xrange
        return self.loss_fx(x, new_min, new_max, tau).cpu().numpy()

class SignAvgMSEFastObserver(AvgMSEFastObserver):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1, sign=None):
        super().__init__(bit, symmetric, ch_axis)
        self.sign = sign
    
    def loss_fx(self, x, new_min, new_max):
        # only consider tensor here
        new_min = torch.tensor(new_min).cuda()
        new_max = torch.tensor(new_max).cuda()
        
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        x_q = fake_quantize_per_tensor_affine(
                    x, scale.item(), int(zero_point.item()),
                    self.quant_min, self.quant_max)

        if self.sign is not None:
            # print(self.sign)
            x_q = x_q*self.sign[None,None,:]
            x = x*self.sign[None,None,:]

        score = self.lp_loss(x_q, x, p=self.p)
        return score

class PCTObserver(ObserverBase):
    # grid search: more accurate but slow
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(PCTObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = 2.0
        self.plist = [0.99,0.999,0.9999,0.99999]
        # self.num = 100  # candidate num
        # self.one_side_dist = None  # 'pos', 'neg', 'no'

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if self.ch_axis == -1:
            return x.mean()
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            return y.mean(1)

    def loss_fx(self, x, new_min, new_max):
        # should also consider channel here
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        if self.ch_axis != -1:
            x_q = fake_quantize_per_channel_affine(
                x, scale.data, zero_point.data.int(), self.ch_axis,
                self.quant_min, self.quant_max)
        else:
            x_q = fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()),
                self.quant_min, self.quant_max)
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def perform_search(self, x):
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            for pct in self.plist:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                # x_q = self.quantize(x_clone, new_max, new_min)
                score = self.loss_fx(x_clone, new_min, new_max)
                if score < best_score:
                    best_min = new_min
                    best_max = new_max
                    # delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    # zero_point = (- new_min / delta).round()
            return best_min, best_max

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)

        best_min, best_max = self.perform_search(x)
        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)

 