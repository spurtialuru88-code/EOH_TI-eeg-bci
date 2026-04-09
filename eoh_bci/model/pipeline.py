from eoh_bci.features.csp import CSPFilter
from eoh_bci.preprocessing.filters import bandpass_filter, notch_filter

class RealTimePipeline:
    def __init__(self, model, csp: CSPFilter):
        self.model = model
        self.csp = csp

    def process(self, signal):
        signal = notch_filter(signal)
        signal = bandpass_filter(signal)

        features = self.csp.transform(signal)
        return self.model.predict([features])[0]