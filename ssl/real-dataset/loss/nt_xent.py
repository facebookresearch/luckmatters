import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, beta, add_one_in_neg, exact_cov, exact_cov_unaug_sim):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.beta = beta
        self.temperature = temperature
        self.add_one_in_neg = add_one_in_neg
        self.exact_cov = exact_cov
        self.exact_cov_unaug_sim = exact_cov_unaug_sim
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.mask_samples_small = self._get_correlated_mask_small().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def need_unaug_data(self):
        return self.exact_cov_unaug_sim

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
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        if self.exact_cov:
            # 1 - sim = dist
            r_neg = 1 - negatives
            r_pos = 1 - positives

            num_negative = negatives.size(1)

            # Similarity matrix for unaugmented data.
            if self.exact_cov_unaug_sim and zs is not None:
                similarity_matrix2 = self.similarity_function(zs, zs)
                negatives_unaug = similarity_matrix2[self.mask_samples_small].view(self.batch_size, -1)
                r_neg_unaug = 1 - negatives_unaug
                w = (-r_neg_unaug.detach() / self.temperature).exp() 
                # Duplicated four times. 
                w = torch.cat([w, w], dim=0)
                w = torch.cat([w, w], dim=1)
            else:
                w = (-r_neg.detach() / self.temperature).exp() 
            
            w = w / (1 + w) / self.temperature / num_negative
            # Then we construct the loss function. 
            w_pos = w.sum(dim=1, keepdim=True)
            loss = (w_pos * r_pos - (w * r_neg).sum(dim=1)).mean()
            loss_intra = self.beta * (w_pos * r_pos).mean()
        else:
            if self.add_one_in_neg:
                all_ones = torch.ones(2 * self.batch_size, 1).to(self.device)
                logits = torch.cat((positives, negatives, all_ones), dim=1)
            else:
                logits = torch.cat((positives, negatives), dim=1)

            logits /= self.temperature

            labels = torch.zeros(2 * self.batch_size).to(self.device).long()
            loss = self.criterion(logits, labels)

            # Make positive strong than negative to trigger an additional term. 
            loss_intra = -positives.sum() * self.beta / self.temperature
            loss /= (1.0 + self.beta) * 2 * self.batch_size
            loss_intra /= (1.0 + self.beta) * 2 * self.batch_size

        return loss, loss_intra
