import torch

class GaussianDiffusion:
    def __init__(self) -> None:
        betas = torch.linspace(-6, 6, 1000)
        self.betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x0)
        sqrt_alphas_bar = self.sqrt_alphas_bar[t]
        sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[t]
        return sqrt_alphas_bar * x0 + sqrt_one_minus_alphas_bar * noise
