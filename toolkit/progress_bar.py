from tqdm import tqdm
import time


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._time()

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            pause_duration = cur_t - self.last_time
            self.start_t += pause_duration
            # shift, don't reset: update() measures its rate sample as
            # now - last_print_t. Resetting to now made that ~0 for the
            # update at the end of any paused step, feeding absurd it/s
            # samples into tqdm's rate EMA (bar showed 1000+ it/s on
            # multi-second steps). Shifting keeps the pre-pause compute
            # time in the sample while still excluding the pause itself.
            self.last_print_t += pause_duration

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)
