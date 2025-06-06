from torch import nn, Tensor
import torch
import torch.nn.functional as F
import numpy as np
def softmax_helper(x): return F.softmax(x, 1)


def build_loss(config, deep_supervision, pool_op_kernel_sizes, ignore_edge=False, is_classfication=False):
    if is_classfication:
        print('Classifiction LOSS: using CE!')
        loss = nn.CrossEntropyLoss()
    else:
        if deep_supervision:
            print('LOSS: using deep supervision!')
            weight = get_weight_factors(len(pool_op_kernel_sizes))
            if ignore_edge:
                print('LOSS: using ignore edge!')
                loss = MultipleOutputLoss3(DC_and_CE_ignore_edge_loss(config), weight)
            else:
                loss = MultipleOutputLoss2(DC_and_CE_loss(config), weight)
        else:
            if ignore_edge:
                print('LOSS: using ignore edge!')
                loss = DC_and_CE_ignore_edge_loss(config)
            else:
                loss = DC_and_CE_loss(config)
    return loss


def get_weight_factors(net_numpool):
    weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
    mask = np.array([True] + [True if i < net_numpool -
                              1 else False for i in range(1, net_numpool)])
    weights[~mask] = 0
    weights = weights / weights.sum()
    return weights



class MultipleOutputLoss3(nn.Module):
    def __init__(self, loss, weight_factors=None):
        super(MultipleOutputLoss3, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y, flag):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0], flag[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i], flag[i])
        return l

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class DC_and_CE_ignore_edge_loss(nn.Module):
    def __init__(self, config, aggregate="sum", weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        super(DC_and_CE_ignore_edge_loss, self).__init__()

        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ignore_edge_kernel_size = config.LOSS.IGNORE_EDGE_KERNEL_SIZE
        self.ce = RobustCrossEntropyLoss()

        self.ignore_label = ignore_label
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper)

    def forward(self, net_output, target, flags):
        # flags:1:MR, 0:CT
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            # Do not calculate loss for edge areas.
            mask = torch.ones_like(target)

        # Generate an expanded mask
        structuring_element = torch.ones(1, 1, self.ignore_edge_kernel_size, self.ignore_edge_kernel_size, self.ignore_edge_kernel_size).to(target.device)
        dilated_mask = F.conv3d((target>0).float(), structuring_element, padding=(self.ignore_edge_kernel_size-1)/2)

        # Determine the mask edge area
        boundary_mask = (dilated_mask > 0) & (dilated_mask < structuring_element.sum().item())
        boundary_mask = boundary_mask.float()
        # The weight of the edge area is 0.
        mask[boundary_mask == 1] = 0
        for flag_idx in range(flags.shape[0]):
            if flags[flag_idx] == 0:
                mask[flag_idx] = 1

        dc_loss = self.dc(net_output, target,
                          loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(
            net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")
        return result

class DC_and_CE_loss(nn.Module):
    def __init__(self, config, aggregate="sum", weight_ce=1, weight_dice=1,
                 log_dice=False):
        super(DC_and_CE_loss, self).__init__()

        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ignore_label = config.TRAIN.SELECT_IMPORT_VOXEL.IGNORE_LABEL
        if self.ignore_label is not None:
            print('IGNORE LABEL: {}'.format(self.ignore_label))

        if config.LOSS.FLARE24_CHANGE_WEIGHT:
            print('USING: FLARE24 WEIGHT LOSS!')
            ce_weight = torch.FloatTensor(14).zero_().cuda() + 1
            for i in range(14):
                if i in [7, 8, 9, 10, 12]:
                    ce_weight[i] = 7
            self.ce = RobustCrossEntropyLoss(weight=ce_weight)
        else:
            if self.ignore_label is not None:
                self.ce = RobustCrossEntropyLoss(reduction='none')
            else:
                self.ce = RobustCrossEntropyLoss()

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = torch.ones_like(target)

        dc_loss = self.dc(net_output, target,
                          loss_mask=mask) if self.weight_dice != 0 else 0
       
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(
            net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            # ce_loss *= mask[:, 0]
            ce_loss = ce_loss*mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()
        
        # print('ce_loss:', ce_loss.item())
        # print('dc_loss:', dc_loss.item())
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")
        return result


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=True, do_bg=False, smooth=1e-5):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))
        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0]
                         for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0]
                         for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0]
                         for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0]
                         for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn
