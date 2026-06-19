import time


class ExamTimer:

    def __init__(self, duration_minutes: int):
        self.duration = duration_minutes * 60
        self.start_time = time.time()

    def remaining_time(self):
        elapsed = time.time() - self.start_time
        return max(0, self.duration - elapsed)

    def is_expired(self):
        return self.remaining_time() <= 0
