from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import utils
import utils_corrs 
import pprint
from collections import Counter

def accumulate(a, v):
    return a + v if a is not None else v

def get_stat(w):
    # return f"min: {w.min()}, max: {w.max()}, mean: {w.mean()}, std: {w.std()}" 
    if isinstance(w, list):
        w = np.array(w)
    elif isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
    return f"len: {w.shape}, min/max: {np.min(w):#.3f}/{np.max(w):#.3f}, mean: {np.mean(w):#.3f}" 


class StatsBase(ABC):
    def __init__(self, teacher, student, label):
        self.label = label
        self.teacher = teacher
        self.student = student

        self.reset()

    def add(self, o_t, o_s, y):
        ''' Input: teacher data o_t, student data o_s, and label y (if that's available) '''
        self.count += self._add(o_t, o_s, y)

    def reset(self):
        self._reset()
        self.count = 0

    def export(self):
        self.results = self._export()
        if self.label != "":
            return { self.label + "_" + k : utils.to_cpu(v) for k, v in self.results.items() }
        else:
            return { k : utils.to_cpu(v) for k, v in self.results.items() }

    def prompt(self):
        return pprint.pformat(self.results, indent=4)

    @abstractmethod
    def _add(self, o_t, o_s, y):
        pass

    @abstractmethod
    def _export(self):
        pass

    @abstractmethod
    def _reset(self):
        pass


class StatsCollector:
    def __init__(self, teacher, student, label=""):
        self.stats = []
        self.teacher = teacher
        self.student = student
        self.label = label

    def add_stat_obj(self, stat : StatsBase):
        self.stats.append(stat)
    
    def add_stat(self, cls_stat, *args, sub_label="", **kwargs):
        ''' Add stat by specifying its class name directly '''
        self.stats.append(cls_stat(self.teacher, self.student, *args, label=sub_label, **kwargs))

    def reset(self):
        for stat in self.stats:
            stat.reset()

    def add(self, o_t, o_s, y):
        for stat in self.stats:
            stat.add(o_t, o_s, y)

    def export(self):
        res = dict()
        for stat in self.stats:
            res.update(stat.export())

        if self.label != "":
            return { self.label + "_" + k : v for k, v in res.items() }
        return res

    def prompt(self):
        prompt = "\n"
        for stat in self.stats:
            this_prompt = stat.prompt()
            if this_prompt != "":
                prompt += this_prompt + "\n"

        return prompt


class StatsHs(StatsBase):
    def __init__(self, teacher, student, label=""):
        super().__init__(teacher, student, label)

    @staticmethod
    def _compute_Hs(net1, output1, net2, output2):
        # Compute H given the current banch.
        sz1 = net1.sizes
        sz2 = net2.sizes
        
        bs = output1["hs"][0].size(0)
        
        assert sz1[-1] == sz2[-1], "the output size of both network should be the same: %d vs %d" % (sz1[-1], sz2[-1])

        H = torch.cuda.FloatTensor(bs, sz1[-1] + 1, sz2[-1] + 1)
        for i in range(bs):
            H[i,:,:] = torch.eye(sz1[-1] + 1).cuda()

        Hs = []
        betas = []

        # Then we try computing the other rels recursively.
        j = len(output1["hs"])
        pre_bns1 = output1["pre_bns"][::-1]
        pre_bns2 = output2["pre_bns"][::-1]

        for pre_bn1, pre_bn2 in zip(pre_bns1, pre_bns2):
            # W: of size [output, input]
            W1t = net1.from_bottom_aug_w(j).t()
            W2 = net2.from_bottom_aug_w(j)

            # [bs, input_dim_net1, input_dim_net2]
            beta = torch.cuda.FloatTensor(bs, W1t.size(0), W2.size(1))
            for i in range(bs):
                beta[i, :, :] = W1t @ H[i, :, :] @ W2
            # H_new = torch.bmm(torch.bmm(W1, H), W2)

            betas.append(beta.mean(0))

            H = beta.clone()
            gate2 = (pre_bn2.detach() > 0).float()
            H[:, :, :-1] *= gate2[:, None, :]

            gate1 = (pre_bn1.detach() > 0).float()
            H[:, :-1, :] *= gate1[:, :, None]
            Hs.append(H.mean(0))
            j -= 1

        return Hs[::-1], betas[::-1]


    def _add(self, o_t, o_s, y):
        # Compute H given the current banch.
        Hs_st, betas_st = StatsHs._compute_Hs(self.student, o_s, self.teacher, o_t)
        Hs_ss, betas_ss = StatsHs._compute_Hs(self.student, o_s, self.student, o_s)

        n_layer = len(Hs_st)

        if len(self.Hs_ss) == 0:
            self.Hs_ss = [None] * n_layer
            self.Hs_st = [None] * n_layer
            self.betas_ss = [None] * n_layer
            self.betas_st = [None] * n_layer

        for k in range(n_layer):
            self.Hs_ss[k] = accumulate(self.Hs_ss[k], Hs_ss[k])
            self.Hs_st[k] = accumulate(self.Hs_st[k], Hs_st[k])

            self.betas_ss[k] = accumulate(self.betas_ss[k], betas_ss[k])
            self.betas_st[k] = accumulate(self.betas_st[k], betas_st[k])

        return 1

    def _export(self):
        output_Hs_ss = [ H / self.count for H in self.Hs_ss ]
        output_Hs_st = [ H / self.count for H in self.Hs_st ]
        output_betas_ss = [ beta / self.count for beta in self.betas_ss ]
        output_betas_st = [ beta / self.count for beta in self.betas_st ]
        return dict(Hs_ss=output_Hs_ss, betas_ss=output_betas_ss, Hs_st=output_Hs_st, betas_st=output_betas_st)

    def _reset(self):
        self.Hs_ss = []
        self.betas_ss = []

        self.Hs_st = []
        self.betas_st = []

    def prompt(self):
        return ""


class StatsCorr(StatsBase):
    def __init__(self, teacher, student, label="", active_nodes=None, cnt_thres=0.9):
        super().__init__(teacher, student, label)

        self.active_nodes = active_nodes
        self.cnt_thres = cnt_thres 

    def _reset(self):
        num_layer = self.teacher.num_hidden_layers() 

        self.inner_prod = [None] * num_layer
        self.sum_t = [None] * num_layer
        self.sum_s = [None] * num_layer
        self.sum_sqr_t = [None] * num_layer
        self.sum_sqr_s = [None] * num_layer

    def _add(self, o_t, o_s, y):
        # Compute correlation. 
        # activation: [bs, #nodes]
        for k, (h_tt, h_ss) in enumerate(zip(o_t["hs"], o_s["hs"])):
            h_t = h_tt.detach()
            h_s = h_ss.detach()
            self.inner_prod[k] = accumulate(self.inner_prod[k], h_t.t() @ h_s)

            self.sum_t[k] = accumulate(self.sum_t[k], h_t.sum(dim=0)) 
            self.sum_s[k] = accumulate(self.sum_s[k], h_s.sum(dim=0)) 

            self.sum_sqr_t[k] = accumulate(self.sum_sqr_t[k], h_t.pow(2).sum(dim=0)) 
            self.sum_sqr_s[k] = accumulate(self.sum_sqr_s[k], h_s.pow(2).sum(dim=0)) 

        return o_t["hs"][0].size(0)

    def _export(self):
        num_layer = self.teacher.num_hidden_layers() 
        n = self.count

        res = []
        eps = 1e-7

        for ts, t_sum, s_sum, t_sqr, s_sqr in zip(self.inner_prod, self.sum_t, self.sum_s, self.sum_sqr_t, self.sum_sqr_s):
            s_avg = s_sum / n
            t_avg = t_sum / n

            ts_centered = ts / n - torch.ger(t_avg, s_avg)

            t_norm = (t_sqr / n - t_avg.pow(2)).add(eps).sqrt()
            s_norm = (s_sqr / n - s_avg.pow(2)).add(eps).sqrt()
            
            corr = ts_centered / t_norm[:,None] / s_norm[None, :]
            
            res.append(corr)

        return dict(corrs=res)

    def prompt(self):
        summary = ""
        for k, corr in enumerate(self.results["corrs"]):
            score = []
            cnts = []

            # Corrs: [num_techer, num_student] at each layer.
            sorted_corrs, indices = corr.sort(1, descending=True)
            num_teacher = corr.size(0)

            for kk in range(num_teacher):
                if self.active_nodes is not None and kk not in self.active_nodes[k]:
                    continue

                s = list(sorted_corrs[kk])
                score.append(s[0].item())
                if self.cnt_thres is not None:
                    cnt = sum([ ss.item() >= self.cnt_thres for ss in s ])
                    cnts.append(cnt)

            summary += f"L{k}: {get_stat(score)}"
            if self.cnt_thres is not None:
                summary += f", MatchCnt[>={self.cnt_thres}]: {get_stat(cnts)}"
            summary += "\n"

        return summary

class StatsGrad(StatsBase):
    def __init__(self, teacher, student, label=""):
        super().__init__(teacher, student, label)

    def _reset(self):
        num_layer = self.teacher.num_layers() 

        self.weight_grad_norms = torch.FloatTensor(num_layer).fill_(0)
        self.bias_grad_norms = torch.FloatTensor(num_layer).fill_(0)

    def _add(self, o_t, o_s, y):
        # Only count gradient in student. 
        model = self.student

        weight_grad_norms = []
        bias_grad_norms = []

        k = 0
        for w in model.ws_linear:
            self.weight_grad_norms[k] += w.weight.grad.norm().item()
            self.bias_grad_norms[k] += w.bias.grad.norm().item()
            k += 1

        self.weight_grad_norms[k] += model.final_w.weight.grad.norm().item()
        self.bias_grad_norms[k] += model.final_w.bias.grad.norm().item()

        return 1

    def _export(self):
        return dict(weight_grad_norms=self.weight_grad_norms / self.count, bias_grad_norms=self.bias_grad_norms / self.count)


class StatsL2Loss(StatsBase):
    def __init__(self, teacher, student, label=""):
        super().__init__(teacher, student, label)

        self.loss = nn.MSELoss().cuda()

    def _reset(self):
        self.sum_loss = 0.0
        self.n = 0

    def _add(self, o_t, o_s, y):
        err = self.loss(o_s["y"].detach(), o_t["y"].detach())
        self.sum_loss += err.item()
        self.n += o_t["y"].size(0)
        return 1

    def _export(self):
        return dict(mse_loss=self.sum_loss / self.count, n=self.n) 


class StatsCELoss(StatsBase):      
    def __init__(self, teacher, student, label=""):
        self.top_n = 5
        super().__init__(teacher, student, label)

        self.loss = nn.CrossEntropyLoss().cuda()

    def _reset(self):
        self.sum_loss_teacher = 0.0
        self.sum_topn_teacher = torch.FloatTensor(self.top_n).fill_(0)
        self.n = 0
        self.n_class = 0
        self.class_counts_teacher = Counter()
        self.class_counts_label = Counter()

        self.label_valid = True
        self.sum_loss_label = 0.0
        self.sum_topn_label = torch.FloatTensor(self.top_n).fill_(0)

    def _get_topn(self, predicted_prob, gt):
        probs, predicted = predicted_prob.sort(1, descending=True)
        topn = torch.FloatTensor(self.top_n)
        for i in range(self.top_n):
            topn[i] = (predicted[:, i] == gt).float().mean().item() * 100
            if i > 0:
                topn[i] += topn[i - 1]

        return topn

    def _add(self, o_t, o_s, y):
        teacher_prob = o_t["y"].detach()
        _, teacher_y = teacher_prob.max(1)

        predicted_prob = o_s["y"].detach()

        err = self.loss(predicted_prob, teacher_y)
        self.sum_loss_teacher += err.item()
        self.sum_topn_teacher += self._get_topn(predicted_prob, teacher_y)

        self.class_counts_teacher.update(teacher_y.tolist())

        if (y < 0).sum() == 0:
            err = self.loss(predicted_prob, y)
            self.sum_loss_label += err.item()
            self.sum_topn_label += self._get_topn(predicted_prob, y)
            self.class_counts_label.update(y.tolist())
        else:
            self.label_valid = False

        self.n += o_s["y"].size(0)
        self.n_class = o_s["y"].size(1)

        return 1

    def _export(self):
        class_distri_teacher = self.class_counts_teacher.most_common()
        results = {
            "n": self.n,
            "n_class": self.n_class,
            "ce_loss_teacher" : self.sum_loss_teacher / self.count, 
            f"top{self.top_n}_teacher": self.sum_topn_teacher / self.count,
            "class_count_distri_teacher": class_distri_teacher 
        }

        if self.label_valid:
            class_distri_label = self.class_counts_label.most_common()
            results.update({
                "ce_loss_label" : self.sum_loss_label / self.count, 
                f"top{self.top_n}_label": self.sum_topn_label / self.count,
                "class_count_distri_label": class_distri_label
            })

        return results

    def prompt(self):
        dicts_print = dict()
        for k, v in self.results.items():
            if k.startswith("class_count_distri") and len(v) > 6:
                dicts_print[k] = f"Most common: {v[:3]}, most rare: {v[-3:]}"
            else:
                dicts_print[k] = v

        return pprint.pformat(dicts_print, indent=4)


class StatsMemory(StatsBase):
    def __init__(self, teacher, student, label=""):
        super().__init__(teacher, student, label)

    def _reset(self):
        pass

    def _add(self, o_t, o_s, y):
        return 1

    def _export(self):
        return dict(memory_usage=utils.get_mem_usage())
