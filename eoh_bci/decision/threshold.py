class ThresholdClassifier:
    def __init__(self, rest, flex, dead_zone=0.05):
        self.rest = rest
        self.flex = flex
        self.dead_zone = dead_zone

    def predict(self, value):
        if value is None:
            return None

        # Dead zone (prevents jitter)
        if (self.rest - self.dead_zone) < value < (self.flex + self.dead_zone):
            return None

        if value <= self.rest:
            return 'R'

        if value >= self.flex:
            return 'F'

        return None

    def decide_action(value, low=0.4, high=0.6):
        """
        Maps EEG value → action
        """

        if value > high:
            return "F"   # FLEX
        elif value < low:
            return "R"   # REST
        else:
            return None  # DEAD ZONE