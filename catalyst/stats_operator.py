from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import utils
import utils_corrs 
import pprint

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

    def _add(self, o_t, o_s, y):
        # Compute H given the current banch.
        sz_t = self.teacher.sizes
        sz_s = self.student.sizes
        
        bs = o_t["hs"][0].size(0)
        
        assert sz_t[-1] == sz_s[-1], \
                f"the output size of teacher/student network should be the same: teacher {sz_t[-1]} vs student {sz_s[-1]}"

        H = torch.cuda.FloatTensor(bs, sz_s[-1] + 1, sz_t[-1] + 1)
        for i in range(bs):
            H[i,:,:] = torch.eye(sz_t[-1] + 1).cuda()

        # Then we try computing the other rels recursively.
        j = len(o_t["hs"])
        pre_bns_t = o_t["pre_bns"][::-1]
        pre_bns_s = o_s["pre_bns"][::-1]

        if len(self.Hs) == 0:
            self.Hs = [None] * len(pre_bns_t)
            self.betas = [None] * len(pre_bns_t)

        for pre_bn_t, pre_bn_s in zip(pre_bns_t, pre_bns_s):
            # W: of size [output, input]
            Ws_t = self.student.from_bottom_aug_w(j).t()
            Wt = self.teacher.from_bottom_aug_w(j)

            # [bs, input_dim_net1, input_dim_net2]
            beta = torch.cuda.FloatTensor(bs, Ws_t.size(0), Wt.size(1))
            for i in range(bs):
                beta[i, :, :] = Ws_t @ H[i, :, :] @ Wt
            # H_new = torch.bmm(torch.bmm(W1, H), W2)

            self.betas[j - 1] = accumulate(self.betas[j - 1], beta.mean(0))

            H = beta.clone()
            gate_t = (pre_bn_t.detach() > 0).float()
            H[:, :, :-1] *= gate_t[:, None, :]

            gate_s = (pre_bn_s.detach() > 0).float()
            H[:, :-1, :] *= gate_s[:, :, None]
            
            self.Hs[j - 1] = accumulate(self.Hs[j - 1], H.mean(0))
            j -= 1

        return 1

    def _export(self):
        output_Hs = [ H / self.count for H in self.Hs ]
        output_betas = [ beta / self.count for beta in self.betas ]
        return dict(Hs=output_Hs, betas=output_betas)

    def _reset(self):
        self.Hs = []
        self.betas = []

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

        if (y < 0).sum() == 0:
            err = self.loss(predicted_prob, y)
            self.sum_loss_label += err.item()
            self.sum_topn_label += self._get_topn(predicted_prob, y)
        else:
            self.label_valid = False

        self.n += o_s["y"].size(0)

        return 1

    def _export(self):
        results = {
            "n": self.n,
            "ce_loss_teacher" : self.sum_loss_teacher / self.count, 
            f"top{self.top_n}_teacher": self.sum_topn_teacher / self.count,
        }

        if self.label_valid:
            results.update({
                "ce_loss_label" : self.sum_loss_label / self.count, 
                f"top{self.top_n}_label": self.sum_topn_label / self.count
            })

        return results


class StatsMemory(StatsBase):
    def __init__(self, teacher, student, label=""):
        super().__init__(teacher, student, label)

    def _reset(self):
        pass

    def _add(self, o_t, o_s, y):
        return 1

    def _export(self):
        return dict(memory_usage=utils.get_mem_usage())
