import torch
from abc import abstractmethod
from typing import Union, Iterable, Optional

Numeric = Union[torch.Tensor, int, float]
Numerics = Union[Iterable[Numeric]]


class Meter:
    def __init__(self, device: Union[str, torch.device] = 'cuda', iterable: Optional[Numerics] = None):
        self.device = torch.device(device)
        if iterable is not None:
            self.addN(iterable)

    @abstractmethod
    def add(self, datum: Numeric):
        pass

    def addN(self, iterable: Numerics):
        for datum in iterable:
            self.add(datum)


class StatsMeter(Meter):
    """
    GPU-accelerated PyTorch version for tracking online statistics:
        - mean
        - std / variance
    Uses Welford's algorithm entirely on GPU.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
    """

    def __init__(self, device: Union[str, torch.device] = 'cuda', iterable: Optional[Numerics] = None):
        """Online Mean and Variance computed entirely on GPU

        Args:
            device: PyTorch device ('cuda', 'cpu', or torch.device object)
            iterable: Values to initialize (default: None)
        """
        self.device = torch.device(device)
        self.n: torch.Tensor = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.mean: torch.Tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.S: torch.Tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        if iterable is not None:
            self.addN(iterable)

    def _to_tensor(self, datum: Numeric) -> torch.Tensor:
        """Convert input to a tensor on the correct device"""
        if isinstance(datum, (int, float)):
            return torch.tensor(datum, dtype=torch.float32, device=self.device)
        elif isinstance(datum, torch.Tensor):
            return datum.to(device=self.device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type: {type(datum)}")

    def add(self, datum: Numeric):
        """Add a single datum using Welford's method on GPU

        Args:
            datum: Numerical value (int, float, or torch.Tensor)
        """
        datum = self._to_tensor(datum)
        
        self.n += 1
        delta = datum - self.mean
        # Mk = Mk-1 + (xk – Mk-1)/k
        self.mean += delta / self.n.float()
        # Sk = Sk-1 + (xk – Mk-1)*(xk – Mk)
        self.S += delta * (datum - self.mean)

    def addN(self, iterable: Numerics, batch: bool = False):
        """Add N data to the stats

        Args:
            iterable: Iterable of numerical values
            batch: If True, compute stats over the batch and merge
        """
        if batch:
            # Convert iterable to tensor
            if isinstance(iterable, torch.Tensor):
                data = iterable.to(device=self.device, dtype=torch.float32)
            else:
                data = torch.tensor(list(iterable), dtype=torch.float32, device=self.device)
            
            # Compute batch statistics
            batch_n = data.numel()
            batch_mean = data.mean()
            batch_std = data.std(unbiased=False)
            
            # Merge with current stats
            add = self + StatsMeter.from_values(batch_n, batch_mean, batch_std, device=self.device)
            self.n, self.mean, self.S = add.n, add.mean, add.S
        else:
            for datum in iterable:
                self.add(datum)

    def pop(self, datum: Numeric):
        """Remove a datum from the statistics"""
        if self.n == 0:
            raise ValueError("Stats must be non empty")

        datum = self._to_tensor(datum)
        
        self.n -= 1
        delta = datum - self.mean
        # Mk-1 = Mk - (xk - Mk) / (k - 1)
        self.mean -= delta / self.n.float()
        # Sk-1 = Sk - (xk – Mk-1) * (xk – Mk)
        self.S -= (datum - self.mean) * delta

    def popN(self, iterable: Numerics, batch: bool = False):
        """Remove N data from the stats"""
        if batch:
            raise NotImplementedError
        else:
            for datum in iterable:
                self.pop(datum)

    @property
    def variance(self) -> torch.Tensor:
        """Sample variance"""
        if self.n == 0:
            return torch.tensor(float('nan'), device=self.device)
        return self.S / self.n.float()

    @property
    def std(self) -> torch.Tensor:
        """Sample standard deviation"""
        return torch.sqrt(self.variance)

    @staticmethod
    def from_values(n: int, mean: Union[float, torch.Tensor], std: Union[float, torch.Tensor], 
                    device: Union[str, torch.device] = 'cuda') -> "StatsMeter":
        """Create StatsMeter from n, mean, and std"""
        stats = StatsMeter(device=device)
        stats.n = torch.tensor(n, dtype=torch.int64, device=stats.device)
        
        if isinstance(mean, torch.Tensor):
            stats.mean = mean.to(device=stats.device, dtype=torch.float32)
        else:
            stats.mean = torch.tensor(mean, dtype=torch.float32, device=stats.device)
        
        if isinstance(std, torch.Tensor):
            std_tensor = std.to(device=stats.device, dtype=torch.float32)
        else:
            std_tensor = torch.tensor(std, dtype=torch.float32, device=stats.device)
        
        stats.S = std_tensor ** 2 * n
        return stats

    @staticmethod
    def from_raw_values(n: int, mean: Union[float, torch.Tensor], S: Union[float, torch.Tensor],
                       device: Union[str, torch.device] = 'cuda') -> "StatsMeter":
        """Create StatsMeter from raw n, mean, and S values"""
        stats = StatsMeter(device=device)
        stats.n = torch.tensor(n, dtype=torch.int64, device=stats.device)
        
        if isinstance(mean, torch.Tensor):
            stats.mean = mean.to(device=stats.device, dtype=torch.float32)
        else:
            stats.mean = torch.tensor(mean, dtype=torch.float32, device=stats.device)
        
        if isinstance(S, torch.Tensor):
            stats.S = S.to(device=stats.device, dtype=torch.float32)
        else:
            stats.S = torch.tensor(S, dtype=torch.float32, device=stats.device)
        
        return stats

    def __str__(self) -> str:
        return f"n={self.n.item()}  mean={self.mean.item():.6f}  std={self.std.item():.6f}"

    def __repr__(self) -> str:
        if self.n == 0:
            return f"{self.__class__.__name__}(device='{self.device}')"
        return (
            f"{self.__class__.__name__}.from_values("
            f"n={self.n.item()}, mean={self.mean.item():.6f}, "
            f"std={self.std.item():.6f}, device='{self.device}')"
        )

    def __add__(self, other: Union[Numeric, "StatsMeter"]) -> "StatsMeter":
        """Add two StatsMeter objects or add a constant

        Args:
            other: Another StatsMeter or a numeric constant

        Returns:
            New StatsMeter with combined statistics
        """
        if isinstance(other, StatsMeter):
            # Merge two independent samples
            n1, n2 = self.n.float(), other.n.float()
            mu1, mu2 = self.mean, other.mean
            S1, S2 = self.S, other.S
            
            # New stats
            n = n1 + n2
            mu = n1 / n * mu1 + n2 / n * mu2
            S = (S1 + n1 * mu1 * mu1) + (S2 + n2 * mu2 * mu2) - n * mu * mu
            
            return StatsMeter.from_raw_values((n1 + n2).int().item(), mu, S, device=self.device)
        elif isinstance(other, (int, float, torch.Tensor)):
            # Add constant to all values (only changes mean)
            other_tensor = self._to_tensor(other)
            return StatsMeter.from_raw_values(
                self.n.item(), self.mean + other_tensor, self.S, device=self.device
            )
        else:
            raise TypeError("Can only add other StatsMeter objects or numbers")

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, k: Union[float, int, torch.Tensor]) -> "StatsMeter":
        """Multiply all values by a constant"""
        k_tensor = self._to_tensor(k)
        return StatsMeter.from_raw_values(
            self.n.item(), self.mean * k_tensor, self.S * k_tensor ** 2, device=self.device
        )

    def asdict(self) -> dict:
        """Return statistics as dictionary with Python scalars"""
        return {
            "mean": self.mean.item() if self.mean.numel() == 1 else self.mean.cpu().numpy(),
            "std": self.std.item() if self.std.numel() == 1 else self.std.cpu().numpy(),
            "n": self.n.item()
        }

    @property
    def flatmean(self) -> torch.Tensor:
        """Mean across all dimensions (for array-valued datapoints)"""
        return self.mean.mean()

    @property
    def flatvariance(self) -> torch.Tensor:
        """Variance across all dimensions (for array-valued datapoints)"""
        return (self.variance + self.mean ** 2).mean() - self.flatmean ** 2

    @property
    def flatstd(self) -> torch.Tensor:
        """Std across all dimensions (for array-valued datapoints)"""
        return torch.sqrt(self.flatvariance)


class MeterDict(dict):
    """Dictionary of meters with automatic creation"""
    
    def __init__(self, meter_type=StatsMeter, device: Union[str, torch.device] = 'cuda'):
        self._meter_type = meter_type
        self.device = torch.device(device)
        super().__init__()

    def update(self, data):
        """Update multiple meters from a dictionary"""
        for label, value in data.items():
            self[label].add(value)

    def __setitem__(self, key, value):
        """Set item creates meter if needed and adds value"""
        if key not in self:
            self[key]  # Create meter
        self[key].add(value)

    def __getitem__(self, key):
        """Get item creates meter if it doesn't exist"""
        if key not in self:
            super().__setitem__(key, self._meter_type(device=self.device))
        return super().__getitem__(key)

    def collect(self, attr):
        """Collect attribute from all meters"""
        return {label: getattr(meter, attr) for label, meter in self.items()}

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(dict(self))})"

    def add(self, label, value):
        """Add value to a specific meter"""
        self[label].add(value)

