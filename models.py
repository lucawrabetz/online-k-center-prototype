class FLFullInstance:
    def __init__(self):
        self.n: int = 1
        self.T: int = 1
        self.x: List[List[float]] = [[0.0]]
        self.Gamma: float = 20.0

class FLInstanceDistributor:
    def __init__(self, full_instance):
        self.full_instance = full_instance
