from scamp import Session
import threading
import queue

class Synth:

    def __init__(self) -> None:
        self.s = Session()
        self.cello = self.s.new_part("Cello")
        self.audio_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()

    def _audio_worker(self):
        """Background thread that plays audio without blocking the main loop."""
        while True:
            try:
                pitch, volume, duration = self.audio_queue.get(timeout=1.0)
                self.cello.play_note(pitch, volume, duration)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")

    def gen_sound(self, energy, presence, animal=None):
        """
        Queue a sound to be played in a background thread (non-blocking).
        
        Args:
            energy: Controls volume (0.0 to 1.0)
            presence: Controls pitch (0.0 to 1.0, mapped to MIDI range)
            animal: Reserved for future use
        """
        pitch = 30 + (presence * 60)
        volume = max(0.0, min(1.0, energy))
        duration = 0.2
        
        try:
            self.audio_queue.put_nowait((pitch, volume, duration))
        except queue.Full:
            pass