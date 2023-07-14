from collections import defaultdict


class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key] for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats
