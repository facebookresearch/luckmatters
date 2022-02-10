import torch
import numpy as np

def quadratic_assignment(dists, tau):
    # each row of dists need to be converged to alpha. 
    N = dists.size(1)
    sorted_dists, sorted_indices = dists.sort(1)
    moving_sum = sorted_dists.cumsum(dim=1)
    indices = torch.arange(0, N).to(moving_sum.device)
    moving_avg = (moving_sum + tau) / (indices[None,:] + 1) 
    indices_to_keep = sorted_dists < moving_avg
    first_indices = torch.logical_xor(indices_to_keep[:, 1:], indices_to_keep[:, :-1]).float() @ indices[1:].float() 
    mu = moving_avg.gather(1, first_indices.long().view(dists.size(0), -1)).squeeze()
    return (mu[:, None] - dists).clamp(min=0) / tau 

def inverse_assignment(dists, tau, inverse_exponent):
    return 1 / (dists + tau).pow(inverse_exponent)

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, params):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        # temperature, use_cosine_similarity, beta, add_one_in_neg, loss_type, exact_cov_unaug_sim
        self.params = params

        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.mask_samples_small = self._get_correlated_mask_small().type(torch.bool)
        self.similarity_function = self._get_similarity_function(params["use_cosine_similarity"])
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def need_unaug_data(self):
        return self.params["exact_cov_unaug_sim"]

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def _get_correlated_mask_small(self):
        diag = np.eye(self.batch_size)
        mask = torch.from_numpy(diag)
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (2N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (2N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, zs):
        # Two towers. For each of the N samples, it has zj and zi. 
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # 2N positive pairs 
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # 2N * (2N - 2) negative samples. 
        # The i-th row corresponds to 2N - 1 negative samples for i-th sample.  
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        temperature = self.params["temperature"]
        beta = self.params["beta"]
        loss_type = self.params["loss_type"]
        alpha_exponent = self.params["alpha_exponent"]
        alpha_type = self.params["alpha_type"]
        inverse_exponent = self.params["inverse_exponent"]

        if loss_type == "exact_cov":
            # 1 - sim = dist
            r_neg = 1 - negatives
            r_pos = 1 - positives

            num_negative = negatives.size(1)

            # Similarity matrix for unaugmented data.
            if self.exact_cov_unaug_sim and zs is not None:
                similarity_matrix2 = self.similarity_function(zs, zs)
                negatives_unaug = similarity_matrix2[self.mask_samples_small].view(self.batch_size, -1)
                r_neg_unaug = 1 - negatives_unaug
                w = (-r_neg_unaug.detach() / temperature).exp() 
                # Duplicated four times. 
                w = torch.cat([w, w], dim=0)
                w = torch.cat([w, w], dim=1)
            else:
                w = (-r_neg.detach() / temperature).exp() 
            
            w = w / (1 + w) / temperature / num_negative
            # Then we construct the loss function. 
            w_pos = w.sum(dim=1, keepdim=True)
            loss = (w_pos * r_pos - (w * r_neg).sum(dim=1)).mean()
            loss_intra = beta * (w_pos * r_pos).mean()

        elif loss_type == "dual":
            # 1 - sim = dist
            r_neg = 1 - negatives
            r_pos = 1 - positives

            w = r_neg.detach().pow(alpha_exponent)
            if alpha_type == "exp":
                w = (-w / temperature).exp()
            elif alpha_type == "quadratic":
                w = quadratic_assignment(w, temperature)
            elif alpha_type == "inverse":
                w = inverse_assignment(w, temperature, inverse_exponent)
            else:
                raise RuntimeError(f"Unknown alpha_type = {alpha_type}")

            # The below is actually mean(w * (r_pos - r_neg))
            w_pos = w.sum(dim=1, keepdim=True)
            loss = (w_pos * r_pos - (w * r_neg).sum(dim=1)).mean()
            loss_intra = beta * (w_pos * r_pos).mean()

        elif loss_type == "dual_backprop":
            # 1 - sim = dist
            r_neg = 1 - negatives
            r_pos = 1 - positives

            w = r_neg.pow(alpha_exponent)
            if alpha_type == "exp":
                w = (-w / temperature).exp()
            # The below is actually mean(w * (r_pos - r_neg))
            w_pos = w.sum(dim=1, keepdim=True)
            loss = (w_pos * r_pos - (w * r_neg).sum(dim=1)).mean()
            loss_intra = beta * (w_pos * r_pos).mean()

        elif loss_type == "dual_lowrank":
            # 1 - sim = dist
            r_neg = 1 - negatives
            r_pos = 1 - positives

            w = (-r_neg.detach() / temperature).exp() 
            # get approximate low rank decomposition. 
            sample_importance = w.mean(dim=1)
            #sample_importance = sample_importance / sample_importance.sum()
            similarity_matrix_low_rank = torch.outer(sample_importance, sample_importance)

            w_low_rank = similarity_matrix_low_rank[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
            w_pos_low_rank = w_low_rank.sum(dim=1, keepdim=True) 

            # The below is actually mean(w_low_rank * (r_pos - r_neg))
            loss = (w_pos_low_rank * r_pos - (w_low_rank * r_neg).sum(dim=1)).mean()
            loss_intra = beta * (w_pos_low_rank * r_pos).mean()

        elif loss_type == "default":
            if self.params["add_one_in_neg"]:
                all_ones = torch.ones(2 * self.batch_size, 1).to(self.device)
                logits = torch.cat((positives, negatives, all_ones), dim=1)
            else:
                logits = torch.cat((positives, negatives), dim=1)

            logits /= temperature

            labels = torch.zeros(2 * self.batch_size).to(self.device).long()
            loss = self.criterion(logits, labels)

            # Make positive strong than negative to trigger an additional term. 
            loss_intra = -positives.sum() * beta / temperature
            loss /= (1.0 + beta) * 2 * self.batch_size
            loss_intra /= (1.0 + beta) * 2 * self.batch_size
        
        elif loss_type == "quadratic":
            loss_intra = -positives.mean()
            loss = negatives.mean()

        else:
            raise RuntimeError(f"Unknown loss_type = {loss_type}")

        return loss, loss_intra, negatives.detach()
