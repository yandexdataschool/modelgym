from tracker import Tracker

class NoneTracker(Tracker):
    def __init__(self):
        pass

    def save_state(self):
        pass

    def get_state(self):
        return None