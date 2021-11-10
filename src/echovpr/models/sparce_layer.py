import torch
import torch.nn as nn
import torch.nn.init as init


class SpaRCe(nn.Module):
    """SpaRCE Layer"""

    in_features: int
    sparce_quantile: torch.Tensor
    sparce_theta_tilde: torch.Tensor

    def __init__(self, in_features, precomputed_sparce_quantile=None):
        super().__init__()
        self.in_features = in_features
        self.register_buffer('sparce_quantile', torch.empty((in_features), dtype=torch.float) if precomputed_sparce_quantile is None else precomputed_sparce_quantile);
        self.sparce_theta_tilde = nn.Parameter(torch.zeros((in_features), dtype=torch.float))
        self.reset_parameters()

    def forward(self, x):
        sparse_output = torch.sign(x) * torch.relu(torch.abs(x) - self.sparce_quantile + self.sparce_theta_tilde)
        return sparse_output

    def forward(self, x, sparce_quantile):
        sparse_output = torch.sign(x) * torch.relu(torch.abs(x) - sparce_quantile + self.sparce_theta_tilde)
        return sparse_output

    def set_sparce_quantile(self, quantile: torch.Tensor):
        self.sparce_quantile.copy_(quantile)

    def reset_parameters(self) -> None:
        init.normal_(self.sparce_theta_tilde, 0, 1)
        with torch.no_grad():
            self.sparce_theta_tilde.divide_(1000)

    def extra_repr(self) -> str:
        return 'in_features={}'.format(
            self.in_features
        )
