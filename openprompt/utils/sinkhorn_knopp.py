import torch


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits, prior_targets=None):
        if prior_targets==None:
            Q = torch.exp(logits / self.epsilon).t()

            B = Q.shape[1]  # 228
            K = Q.shape[0]  # how many prototypes  31

            # make the matrix sums to 1
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            for it in range(self.num_iters):
                # normalize each row: total weight per prototype must be 1/K
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_of_rows
                Q /= K

                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B

            Q *= B  # the colomns must sum to 1 so that Q is an assignment

            return Q.t()

        else:
            prior_distribution = torch.sum(prior_targets, dim=0)
            prior_distribution = torch.clamp(prior_distribution, min=1)
            Q = torch.exp(logits / self.epsilon).t()

            B = Q.shape[1]
            K = Q.shape[0]  # how many prototypes

            # make the matrix sums to 1
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            for it in range(self.num_iters):
                # normalize each row: total weight per prototype must be 1/K
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_of_rows
                for i in range(K):
                    Q[i,:] = Q[i,:] * (prior_distribution[i])
                Q /= prior_distribution.sum()

                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B

            Q *= B  # the colomns must sum to 1 so that Q is an assignment
            return Q.t()
