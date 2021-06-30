import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import _create_model_training_folder
from copy import deepcopy

import logging
log = logging.getLogger(__file__)

def check_valid(M):
    if torch.any(torch.logical_or(torch.isnan(M), torch.isinf(M))).item():
        import pdb
        pdb.set_trace()

def random_orthonormal_matrix(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

def weights_init(m, params):
    torch.nn.init.xavier_uniform_(m.weight.data)
    # compute min/max
    l = params["low"]
    h = params["high"]
    if l is None and h is not None:
        l = -h
    if h is None and l is not None:
        h = -l
        
    log.info(f"xavier min: {m.weight.data.min().item()}, max: {m.weight.data.max()}")
    if l is not None and h is not None:
        log.info(f"New range: [{l}, {h}]")
        torch.nn.init.uniform_(m.weight.data, a=l, b=h)

    if m.bias is not None:
        m.bias.data.fill_(0)

class Accumulator:
    def __init__(self, dyn_lambda=None):
        self.dyn_lambda = dyn_lambda
        self.cumulated = None
        self.counter = 0

        self.reset()

    def reset(self):
        if self.dyn_lambda is None:
            # Averaging..
            self.cumulated = None
            self.counter = 0

    def add_list(self, d_list):
        assert isinstance(d_list, list)

        all_d = torch.cat(d_list, dim=0)
        if all_d.size(0) == 0:
            d = torch.zeros(*all_d.size()[1:]).to(device=all_d.get_device())
        else:
            d = all_d.mean(dim=0)

        self.add(d)

    def add(self, d):
        if self.cumulated is None:
            self.cumulated = d
        else:
            if self.dyn_lambda is None:
                self.cumulated += d
            else:
                self.cumulated = self.dyn_lambda * self.cumulated + (1 - self.dyn_lambda) * d

        self.counter += 1

    def get(self):
        if self.dyn_lambda is None:
            assert self.counter > 0
            return self.cumulated / self.counter
        else:
            return self.cumulated.clone()

class BYOLTrainer:
    def __init__(self, log_dir, online_network, target_network, predictor, optimizer, predictor_optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.predictor_optimizer = predictor_optimizer
        self.device = device
        self.predictor = predictor
        self.params = params
        self.writer = SummaryWriter(log_dir)

        self.rand_pred_n_epoch = params["rand_pred_n_epoch"]
        self.rand_pred_n_iter = params["rand_pred_n_iter"]
        self.rand_pred_reg = params["rand_pred_reg"]
        self.max_epochs = params['max_epochs']
        self.m = params['m']
        self.use_order_of_variance = params["use_order_of_variance"]
        self.corr_eigen_decomp = params["corr_eigen_decomp"]
        self.noise_blend = params["noise_blend"]
        self.save_per_epoch = params["save_per_epoch"]
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.target_noise = params['target_noise']
        self.predictor_init = params["predictor_init"]
        self.predictor_reg = params["predictor_reg"]
        self.predictor_eig = params["predictor_eig"]
        self.predictor_freq = params["predictor_freq"]
        self.predictor_rank = params["predictor_rank"]
        self.predictor_eps = params["predictor_eps"]
        self.dyn_time = params["dyn_time"]
        self.dyn_zero_mean = params["dyn_zero_mean"]
        self.dyn_reg = params["dyn_reg"]
        self.dyn_noise = params["dyn_noise"]
        self.dyn_lambda = params["dyn_lambda"]
        self.dyn_sym = params["dyn_sym"]
        self.dyn_psd = params["dyn_psd"]
        self.dyn_eps = params["dyn_eps"]
        self.dyn_eps_inside = params["dyn_eps_inside"]
        self.dyn_bn = params["dyn_bn"]
        self.dyn_convert = params["dyn_convert"]
        self.dyn_diagonalize = params["dyn_diagonalize"]
        self.balance_type = params["balance_type"]
        self.evaluator = params["evaluator"]
        self.solve_direction = params["solve_direction"]
        self.corr_collect = params["corr_collect"]
        self.n_corr = params["n_corr"]
        self.use_l2_normalization = params["use_l2_normalization"]
        self.predictor_wd = params["predictor_wd"]
        self.init_rand_pred = params["init_rand_pred"]
        _create_model_training_folder(self.writer, files_to_same=["./config/byol_config.yaml", "main.py", 'byol_trainer.py',
                                                                  "./models/mlp_head.py"])

        self.predictor_signaling = False
        self.predictor_signaling_2 = False

        self.cum_corr = Accumulator(dyn_lambda=self.dyn_lambda)
        self.cum_cross_corr = Accumulator(dyn_lambda=self.dyn_lambda)
        self.cum_mean1 = Accumulator(dyn_lambda=self.dyn_lambda)
        self.cum_mean2 = Accumulator(dyn_lambda=self.dyn_lambda)

        if self.dyn_noise is not None:
            self.skew = torch.randn(128, 128).to(device=device)
            self.skew = (self.skew - self.skew.t()) * self.dyn_noise

        if self.predictor_reg == "partition":
            # random partition.
            self.partition_w = torch.randn(128, self.n_corr).to(device=device)
            # accumulate according to random partitions. 
            self.cum_corrs_pos = [Accumulator(dyn_lambda=self.dyn_lambda) for i in range(self.n_corr)]
            self.cum_corrs_neg = [Accumulator(dyn_lambda=self.dyn_lambda) for i in range(self.n_corr)]

            self.counts_pos = [0 for i in range(self.n_corr)]
            self.counts_neg = [0 for i in range(self.n_corr)]

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        # for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
        #     param_k.data = param_q.data + torch.normal(0., self.target_noise * torch.norm(param_q), size=param_q.shape, device='cuda').data

    @staticmethod
    def regression_loss(x, y, l2_normalized=True):
        if l2_normalized:
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
            return -2 * (x * y).sum(dim=-1)
        else:
            # No normalization.
            return (x - y).pow(2).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def get_pred_linear_layers(self):
        # Extract linear layer. 
        self.linear_layers = []
        for module in self.predictor.layers:
            if isinstance(module, nn.Linear):
                self.linear_layers.append(module)

        if not self.predictor_signaling:
            if self.predictor_reg is not None:
                log.info(f"Enforcing {self.predictor_reg} matrix, #linear layer: {len(self.linear_layers)}, freq: {self.predictor_freq}, eig: {self.predictor_eig}, rank: {self.predictor_rank}") 
            if self.rand_pred_n_epoch > 1:
                log.info(f"Randomize: {self.rand_pred_reg}, per_epoch: {self.rand_pred_n_epoch}")
            log.info(self.predictor)
            self.predictor_signaling = True

    def random_init_predictor(self, predictor):
        # Extract linear layer. 
        name2idx = dict(all=[0,1], top=[1], bottom=[0])
        assert self.rand_pred_reg in name2idx, f"Randomized reg {self.rand_pred_reg} is not valid!"
        selected_layers = name2idx[self.rand_pred_reg]

        for i in selected_layers:
            if i < len(self.linear_layers):
                weights_init(self.linear_layers[i], self.predictor_init)

    def noise_blending(self, network, alpha):
        if alpha > 0:
            log.info(f"Noise blending: {alpha}")
            # Also blend the noise in.
            for _, p in network.named_parameters():
                with torch.no_grad():
                    if len(p.size()) < 2:
                        continue
                    w = torch.zeros_like(p, device=p.get_device())
                    torch.nn.init.xavier_uniform_(w)
                    p[:] = (1 - alpha) * p[:] + alpha * w

    def compute_w_corr(self, M):
        if self.corr_eigen_decomp:
            if not self.predictor_signaling_2:
                log.info("compute_w_corr: Use eigen_decomp!")
            D, Q = torch.eig(M, eigenvectors=True)
            # Only use the real part. 
            D = D[:,0]
        else:
            # Just use diagonal element. 
            if not self.predictor_signaling_2:
                log.info("compute_w_corr: No eigen_decomp, just use diagonal elements!")
            D = M.diag()
            Q = torch.eye(M.size(0)).to(D.device)
             
        # if eigen_values >= 1, scale everything down. 
        balance_type = self.balance_type
        reg = self.dyn_reg

        if balance_type == "shrink2mean":
            mean_eig = D.mean()
            eigen_values = (D - mean_eig) / 2 + mean_eig
        elif balance_type == "clamp":
            eigen_values = D.clamp(min=0, max=1-reg)
        elif balance_type == "boost_scale":
            max_eig = D.max()
            eigen_values = D.clamp(0) / max_eig
            # Going through a concave function (dyn_convert > 1, e.g., 2 or sqrt function) to boost small eigenvalues (while still keep very small one to be 0)
            if self.dyn_eps_inside:
                # Note that here dyn_eps is allowed to be negative.
                eigen_values = (eigen_values + self.dyn_eps).clamp(1e-4).pow(1/self.dyn_convert)
            else:
                # Note that here dyn_eps is allowed to be negative.
                eigen_values = eigen_values.pow(1/self.dyn_convert) + self.dyn_eps
                eigen_values = eigen_values.clamp(1e-4)
            if not self.predictor_signaling_2:
                sorted_values, _ = eigen_values.sort(descending=True)
                log.info(f"Compute eigenvalues with boost_scale: Top-5: {sorted_values[:5]}, Bottom-5: {sorted_values[-5:]}")

        elif balance_type == "scale":
            max_eig = D.max()
            if max_eig > 1 - reg:
                eigen_values = D / (max_eig + reg)
        else:
            raise RuntimeError(f"Unkonwn balance_type: {balance_type}")

        return Q @ eigen_values.diag() @ Q.t()

    def compute_w_minimal_space(self, M, M2, w): 
        try:
            # QDQ^t = M
            D, Q = torch.eig(M, eigenvectors=True)
        except RuntimeError:
            import pdb
            pdb.set_trace()
        # The equation is \dot W = -MW - WM + 2 * M2 (when W keeps symmetric)
        # The solution is e^{-Mt} M2 e^{-Mt}
        # If M can be decomposed: M = QDQ^T, then the solution is Qe^{-Dt}Q^T M2 Q e^{-Dt} Q^T
        d = w.size(1)
        eigen_values = D[:,0].clamp(0)
        M2_convert = Q.t() @ M2 @ Q 
        already_psd = False 
        if self.dyn_time is None:
            # integrate things out and you get a matrix which is 2 / (d_i + d_j), where d_i is the eigenvalues. 
            if self.dyn_diagonalize: 
                M2_diag = M2_convert.diag() / (eigen_values + self.dyn_reg / 2)
                if self.dyn_psd is not None:
                    M2_diag.clamp_(0)
                    already_psd = True
                M2_convert = M2_diag.diag()
            else:
                M2_convert /= (eigen_values.view(d, 1) + eigen_values.view(1, d) + self.dyn_reg) / 2
        else:
            eD = (eigen_values * (-self.dyn_time)).exp().diag()
            M2_convert = eD @ M2_convert @ eD

        w = Q @ M2_convert @ Q.t()

        # Project the weight to be PSD?
        # If we choose to make it PSD (or diagonalized), it will be symmetric by default.
        if self.dyn_psd is None:
            return w
        
        if already_psd:
            w_half = Q @ M2_diag.sqrt().diag()
        else:
            D2, Q2 = torch.eig(w, eigenvectors=True)
            eigen_values2 = D2[:,0].clamp(0)

            # w = w_half @ w_half.t() 
            w_half = Q2 @ eigen_values2.sqrt().diag() 

        if self.dyn_psd > 0:
            # then we do a few iterations to make w more precise.
            alpha = 0.01
            err_magnitudes = torch.zeros(self.dyn_psd)
            w_half_hist = []
            for kk in range(self.dyn_psd):
                w = w_half @ w_half.t()
                err = - (M @ w + w @ M) + M2 * 2
                grad = err @ w_half
                err_magnitudes[kk] = grad.norm()
                w_half_hist.append(w_half.clone())
                # print(f"[{kk}] err: {err.norm().item()}, grad: {grad.norm().item()}")
                w_half += alpha * grad

            err_magnitudes[torch.isnan(err_magnitudes)] = 1e38
            best_kk = err_magnitudes.argmin().item()
            w_half = w_half_hist[best_kk]
        w = w_half @ w_half.t()
        return w

    def regulate_predictor(self, predictor, niter, epoch_start=False):
        if self.predictor_reg is None:
            return

        linear_layers = self.linear_layers
        if len(linear_layers) == 1:
            assert self.predictor_reg in ["diagonal", "symmetric", "symmetric_norm", "onehalfeig", "minimal_space", "solve", "corr", "partition"], f"predictor_reg: {self.predictor_reg} not valid!" 
            # Make it symmetric
            with torch.no_grad():
                w = linear_layers[0].weight.clone()
                if torch.any(torch.isnan(w)).item() or torch.any(torch.isinf(w)).item():
                    import pdb
                    pdb.set_trace()
                prev_w = w.clone()

                if self.predictor_reg == "diagonal":
                    # Further make it diagonal
                    w = w.diag().diag()
                elif self.predictor_reg == "symmetric":
                    w += w.t()
                    w /= 2
                elif self.predictor_reg == "symmetric_norm":
                    if not self.predictor_signaling_2:
                        log.info(f"Enforce symmetric constraint with unit spectral norm.")
                    w += w.t()
                    w /= 2
                    # Normalize so that the largest positive eigenvalue is 1
                    D, _ = torch.eig(w, eigenvectors=False)
                    max_eigen = D[:,0].max()
                    if max_eigen.abs() > 1e-2:
                        w /= max_eigen
                    
                elif self.predictor_reg == "solve":
                    if self.predictor_freq > 0 and niter % self.predictor_freq == 0:
                        M = self.cum_corr.get()
                        M2 = self.cum_cross_corr.get()

                        if M is not None and M2 is not None:
                            if not self.predictor_signaling_2:
                                log.info(f"Reinitialize predictor weight (assymmetric). freq={self.predictor_freq}, reg={self.dyn_reg}, dir={self.solve_direction}")

                            D, Q = torch.eig(M, eigenvectors=True)
                            inv = Q @ (D[:,0].clamp(0) + self.dyn_reg).pow(-1) @ Q.t()
                            if self.solve_direction == "left":
                                w = inv @ M2
                            else:
                                w = M2 @ inv

                elif self.predictor_reg == "corr":
                    if self.predictor_freq > 0 and niter % self.predictor_freq == 0:
                        M = self.cum_corr.get()
                        if M is not None:
                            if not self.predictor_signaling_2:
                                log.info(f"Set predictor to align with input correlation. zero_mean={self.dyn_zero_mean}, freq={self.predictor_freq}, type={self.balance_type}, pow=1/{self.dyn_convert}, eps={self.dyn_eps}, reg={self.dyn_reg}, noise={self.dyn_noise}, eps_inside={self.dyn_eps_inside}")

                            if self.dyn_zero_mean:
                                mean_f = self.cum_mean1.get()
                                M -= torch.ger(mean_f, mean_f)

                            w = self.compute_w_corr(M)

                            if self.dyn_noise is not None:
                                w += self.skew / (niter + 1)

                elif self.predictor_reg == "directcopy":
                    if self.predictor_freq > 0 and niter % self.predictor_freq == 0:
                        M = self.cum_corr.get()
                        if M is not None:
                            if not self.predictor_signaling_2:
                                log.info(f"Set predictor to be input correlation. zero_mean={self.dyn_zero_mean}, freq={self.predictor_freq}, eps={self.dyn_eps}")

                            if self.dyn_zero_mean:
                                mean_f = self.cum_mean1.get()
                                M -= torch.ger(mean_f, mean_f)

                            w = M + self.dyn_eps * torch.eye(M.size(0), dtype=M.dtype, device=M.device)

                            if self.dyn_noise is not None:
                                w += self.skew / (niter + 1)

                elif self.predictor_reg == "minimal_space":
                    if self.predictor_freq > 0 and niter % self.predictor_freq == 0:
                        M = self.cum_corr.get()
                        M2 = self.cum_cross_corr.get()

                        if M is not None and M2 is not None:
                            # Initialize weight to contain eigenvectors that correspond to lowest few eigenvalues from the input data.
                            if not self.predictor_signaling_2:
                                log.info(f"Reinitialize predictor weight. freq={self.predictor_freq}, dyn: time={self.dyn_time}, zero_mean={self.dyn_zero_mean}, reg={self.dyn_reg}, sym={self.dyn_sym}, lambda={self.dyn_lambda}, make_psd={self.dyn_psd}, diagonize={self.dyn_diagonalize}, before_bn={self.dyn_bn}")

                            if self.dyn_zero_mean:
                                mean_f = self.cum_mean1.get()
                                mean_f_ema = self.cum_mean2.get()

                                M -= torch.ger(mean_f, mean_f)
                                M2 -= torch.ger(mean_f_ema, mean_f)

                            if self.dyn_sym:
                                M2 += M2.t()
                                M2 /= 2

                            w = self.compute_w_minimal_space(M, M2, w)

                            self.cum_corr.reset()
                            self.cum_cross_corr.reset()
                            self.cum_mean1.reset()
                            self.cum_mean2.reset()
                    else:
                        if self.dyn_sym:
                            # Just make it symmetric
                            w += w.t()
                            w /= 2
                    
                elif self.predictor_reg == "onehalfeig":
                    # only run it with epoch_start 
                    if epoch_start or (self.predictor_freq > 0 and niter % self.predictor_freq == 0):
                        if not self.predictor_signaling_2 or epoch_start:
                            log.info(f"Reinit predictor weight with {self.predictor_reg}. epoch_start={epoch_start}, freq={self.predictor_freq}")
                        d = w.size(0)
                        Q = torch.FloatTensor(random_orthonormal_matrix(dim=d))
                        # Pick predictor_rank of the vectors.
                        w.fill_(0)
                        r = int(self.predictor_rank * d)
                        for i in range(r):
                            w += Q[:,i] @ Q[:,i].t() 
                        w *= self.predictor_eig
                        w += torch.eye(d).to(w.get_device()) * self.predictor_eps
                    else:
                        # Just make it symmetric
                        w += w.t()
                        w /= 2

                elif self.predictor_reg == "partition":
                    # Don't need to do anything. 
                    pass
                else:
                    raise RuntimeError(f"Unknown partition_reg: {self.partition_reg}")

                if torch.any(torch.isnan(w)).item() or torch.any(torch.isinf(w)).item():
                    import pdb
                    pdb.set_trace()

                linear_layers[0].weight.copy_(w)

        elif len(linear_layers) == 2:
            # Two layer without batch norm. Make it symmetric!
            assert self.predictor_reg in ["symmetric", "symmetric_row_norm"], f"predictor_reg: {self.predictor_reg} not valid!"  
            with torch.no_grad():
                # hidden_size x input_size
                w0 = linear_layers[0].weight.clone()
                w1 = linear_layers[1].weight.clone()
                w = (w0 + w1.t()) / 2

                if self.predictor_reg == "symmetric_row_norm":
                    # Keep norm for each hidden layer output.  
                    w /= w.norm(dim=1, keepdim=True)
                    
                linear_layers[0].weight.copy_(w)
                linear_layers[1].weight.copy_(w.t())

        if self.predictor_wd is not None:
            if not self.predictor_signaling_2:
               log.info(f"Apply predictor weight decay: {self.predictor_wd}")

            with torch.no_grad():
                for l in linear_layers:
                    l.weight *= (1 - self.predictor_wd)
            
        self.predictor_signaling_2 = True

    def restart_predictor(self, epoch_start, niter):
        self.random_init_predictor(self.predictor)
        self.regulate_predictor(self.predictor, niter, epoch_start=epoch_start)
        # self.predictor.apply(weights_init)
        log.info(f"Re-initialized predictor. per_epoch: {self.rand_pred_n_epoch}, per_niter: {self.rand_pred_n_iter}")

    def restart_pred_save(self, model_checkpoints_folder, suffix, niter):
        self.restart_predictor(True, niter)
        self.noise_blending(self.online_network, self.noise_blend)

        predictor_path = os.path.join(model_checkpoints_folder, f'reset_predictor_{suffix}_iter{niter}.pth')
        torch.save({'predictor_state_dict': self.predictor.state_dict()}, predictor_path)

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size * torch.cuda.device_count(),
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.get_pred_linear_layers()
        if self.rand_pred_n_epoch is not None and self.rand_pred_n_epoch > 0: 
            self.random_init_predictor(self.predictor)

        if self.init_rand_pred:
            log.info("init_rand_pred=True, reinit the predictor before training starts.")
            self.restart_pred_save(model_checkpoints_folder, "000", 0)

        # Add another BN right before predictor (and right after the target network.)
        self.bn_before_online = nn.BatchNorm1d(self.linear_layers[0].weight.size(1), affine=False).to(self.device)
        self.bn_before_target = nn.BatchNorm1d(self.linear_layers[0].weight.size(1), affine=False).to(self.device)

        # self.initializes_target_network()
        # Save initial network for analysis
        self.save_model(os.path.join(model_checkpoints_folder, 'model_000.pth'))

        for epoch_counter in range(1, 1 + self.max_epochs):
            loss_record = []
            suffix = str(epoch_counter).zfill(3)

            if self.rand_pred_n_epoch is not None and self.rand_pred_n_epoch > 0 and epoch_counter % self.rand_pred_n_epoch == 0:
                self.restart_pred_save(model_checkpoints_folder, suffix, niter)

            for (batch_view_1, batch_view_2, _), _ in train_loader:
                if self.rand_pred_n_iter is not None and self.rand_pred_n_iter > 0 and niter % self.rand_pred_n_iter == 0:
                    self.restart_predictor(False, niter)
                    predictor_path = os.path.join(model_checkpoints_folder, f'reset_predictor_{suffix}_iter{niter}.pth')
                    torch.save({'predictor_state_dict': self.predictor.state_dict()}, predictor_path)

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                self.predictor_optimizer.zero_grad()
                loss.backward()
                # Add additional grad for regularization, if there is any. 
                self.online_network.projetion.adjust_grad()
                self.optimizer.step()
                self.predictor_optimizer.step()

                # self.online_network.projetion.normalize()

                self.regulate_predictor(self.predictor, niter, epoch_start=False)

                self._update_target_network_parameters()  # update the key encoder
                loss_record.append(loss.item())
                niter += 1

            # Reset the signal so that we can print out some statistics. 
            self.predictor_signaling_2 = False

            log.info(f"Epoch {epoch_counter}: numIter: {niter} Loss: {np.mean(loss_record)}")
            if self.evaluator is not None:
                best_acc = self.evaluator.eval_model(deepcopy(self.online_network))
                log.info(f"Epoch {epoch_counter}: best_acc: {best_acc}")

            stats = self.online_network.projetion.get_stats()
            if stats is not None:
                log.info(f"New normalization stats: {stats}")

            if epoch_counter % self.save_per_epoch == 0:
                # save checkpoints
                self.save_model(os.path.join(model_checkpoints_folder, f'model_{suffix}.pth'))
                if self.cum_corr.cumulated is not None:
                    # Save it 
                    corr_path = os.path.join(model_checkpoints_folder, f'corr_{suffix}_iter{niter}.pth')
                    torch.save(
                        {
                            "cumulated_corr": self.cum_corr.cumulated,
                            "cumulated_corr_counter": self.cum_corr.counter,
                            "cumulated_cross_corr": self.cum_cross_corr.cumulated,
                            "cumulated_mean": self.cum_mean1.cumulated,
                            "cumulated_mean_ema": self.cum_mean2.cumulated
                        },
                        corr_path
                    )

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model_final.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        if self.predictor:
            before_predictor_1 = self.online_network(batch_view_1)
            before_predictor_2 = self.online_network(batch_view_2)

            if self.dyn_bn:
               before_predictor_1 = self.bn_before_online(before_predictor_1)
               before_predictor_2 = self.bn_before_online(before_predictor_2)

            if self.predictor_reg in ["minimal_space", "solve", "corr", "partition"] or self.corr_collect:
                corrs = []
                means = []

                corrs_per_partition_pos = [list() for i in range(self.n_corr)]
                corrs_per_partition_neg = [list() for i in range(self.n_corr)]
                partitions = []

                before_detach_1 = before_predictor_1.detach() 
                before_detach_2 = before_predictor_2.detach()

                for b in (before_detach_1, before_detach_2):
                    corr = torch.bmm(b.unsqueeze(2), b.unsqueeze(1))
                    corrs.append(corr)
                    means.append(b)

                    if self.predictor_reg == "partition":
                        pred = b @ self.partition_w
                        thres = pred.mean(dim=0, keepdim=True)
                        # n_batch x n_corr
                        partition = pred >= thres
                        partitions.append(partition)
                        for i in range(self.n_corr):
                            corrs_per_partition_pos[i].append(corr[partition[:,i], :, :])
                            corrs_per_partition_neg[i].append(corr[~partition[:,i], :, :])
                            cnt = partition[:,i].sum().item()
                            self.counts_pos[i] += cnt
                            self.counts_neg[i] += partition.size(0) - cnt

                    if torch.any(torch.isnan(corr)).item():
                        import pdb
                        pdb.set_trace()

                self.cum_mean1.add_list(means)
                self.cum_corr.add_list(corrs)

                if self.predictor_reg == "partition":
                    if not self.predictor_signaling_2:
                        log.info(f"Partition corr matrix. n_corr: {self.n_corr}, counts_pos: {self.counts_pos}, counts_neg: {self.counts_neg}")
                    for i in range(self.n_corr):
                        self.cum_corrs_pos[i].add_list(corrs_per_partition_pos[i])
                        self.cum_corrs_neg[i].add_list(corrs_per_partition_neg[i])
            
            if self.predictor_reg != "partition":
                predictions_from_view_1 = self.predictor(before_predictor_1)
                predictions_from_view_2 = self.predictor(before_predictor_2)
            else:
                # Using special way to compute through predictor.
                ws_pos = torch.zeros(self.n_corr, 128, 128).to(device=before_predictor_1.get_device())
                ws_neg = torch.zeros(self.n_corr, 128, 128).to(device=before_predictor_1.get_device())
                for i in range(self.n_corr):
                    M_pos = self.cum_corrs_pos[i].get()
                    M_neg = self.cum_corrs_neg[i].get()

                    check_valid(M_pos)
                    check_valid(M_neg)

                    ws_pos[i,:,:] = self.compute_w_corr(M_pos)
                    ws_neg[i,:,:] = self.compute_w_corr(M_neg)

                # Then we can use ws according to partition.
                ws_sel_1 = (partitions[0].float() @ ws_pos.view(self.n_corr, -1) + (1 - partitions[0].float()) @ ws_neg.view(self.n_corr, -1)) / self.n_corr
                predictions_from_view_1 = torch.bmm(ws_sel_1.view(-1, 128, 128), before_predictor_1.unsqueeze(2)).squeeze(2)

                ws_sel_2 = (partitions[1].float() @ ws_pos.view(self.n_corr, -1) + (1 - partitions[1].float()) @ ws_neg.view(self.n_corr, -1)) / self.n_corr
                predictions_from_view_2 = torch.bmm(ws_sel_2.view(-1, 128, 128), before_predictor_2.unsqueeze(2)).squeeze(2)
        else:
            predictions_from_view_1 = self.online_network(batch_view_1)
            predictions_from_view_2 = self.online_network(batch_view_2)

        # compute key features
        with torch.no_grad():
            self.target_network.projetion.set_adj_grad(False)

            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

            self.target_network.projetion.set_adj_grad(True)

            if self.dyn_bn:
               targets_to_view_2 = self.bn_before_target(targets_to_view_2)
               targets_to_view_1 = self.bn_before_target(targets_to_view_1)

            if self.predictor_reg in ["minimal_space", "solve", "corr", "directcopy"] or self.corr_collect or self.use_order_of_variance:
                cross_corr1 = torch.bmm(targets_to_view_1.unsqueeze(2), before_detach_1.unsqueeze(1)).mean(dim=0)
                cross_corr2 = torch.bmm(targets_to_view_2.unsqueeze(2), before_detach_2.unsqueeze(1)).mean(dim=0)
                cross_corr = (cross_corr1 + cross_corr2) / 2

                mean_f_ema = (targets_to_view_1.mean(dim=0) + targets_to_view_2.mean(dim=0)) / 2

                if torch.any(torch.isnan(cross_corr)).item():
                    import pdb
                    pdb.set_trace()

                self.cum_mean2.add(mean_f_ema)
                self.cum_cross_corr.add(cross_corr)

        if self.use_order_of_variance:
            if not self.predictor_signaling_2:
                log.info(f"Use order of variance!")

            # Skip the predictor completely.
            M = self.cum_corr.get()
            M2 = self.cum_cross_corr.get()
            # just check their diagonal.
            Mdiag = M.diag()
            M2diag = M2.diag()
            # ratio 
            var_ratio = M2diag / (Mdiag + 1e-5)
            # Ideally we want to have low variance in M2 but high variance in M
            _, indices = var_ratio.sort()
            # Then setup the goal
            d = indices.size(0)
            d_partial = d // 3
            good_indices = indices[:d_partial]
            bad_indices = indices[d_partial:]

            # Compute variance.
            before_predictor = torch.cat([before_predictor_1, before_predictor_2], dim=0)
            before_predictor_normalized = before_predictor / before_predictor.norm(dim=1, keepdim=True)
            variances = before_predictor_normalized.var(dim=0)
            # Minimize the bad variance (suppress the features), while maximize the good variance (boost the feature)
            loss = variances[bad_indices].mean() - variances[good_indices].mean()
        else:
            loss = self.regression_loss(predictions_from_view_1, targets_to_view_1, l2_normalized=self.use_l2_normalization)
            loss += self.regression_loss(predictions_from_view_2, targets_to_view_2, l2_normalized=self.use_l2_normalization)

        return loss.mean()

    def save_model(self, PATH):
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'predictor_optimizer_state_dict': self.predictor_optimizer.state_dict(),
        }, PATH)
