"""Track authenticated browser connections, debounce refresh, and honor explicit exit choices."""
import logging
import os
import threading
import uuid


class Lifecycle:
    def __init__(self, shutdown, prompt=None, delay=3):
        self.shutdown = shutdown
        self.prompt = prompt or self._native_prompt
        self.delay = delay
        self.lock = threading.RLock()
        self.clients = set()
        self.background = False
        self.stopping = False
        self.generation = 0
        self.timer = None
        self.cancel_prompt = threading.Event()

    @staticmethod
    def _native_prompt(cancelled):
        if os.name != 'nt':
            return 'background'
        from .close_dialog import show_close_dialog
        return show_close_dialog(cancelled)

    def connect(self):
        with self.lock:
            if self.stopping:
                raise ValueError('Chep is shutting down.')
            self._invalidate()
            self.background = False
            client = uuid.uuid4().hex
            self.clients.add(client)
            return client

    def disconnect(self, client):
        with self.lock:
            self.clients.discard(client)
            if self.clients or self.background or self.stopping:
                return
            self._invalidate()
            generation = self.generation
            cancelled = self.cancel_prompt = threading.Event()
            self.timer = threading.Timer(self.delay, self._ask, args=(generation, cancelled))
            self.timer.daemon = True
            self.timer.start()

    def _invalidate(self):
        self.generation += 1
        self.cancel_prompt.set()
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def _ask(self, generation, cancelled):
        with self.lock:
            if cancelled.is_set() or self.clients or self.stopping:
                return
        try:
            choice = self.prompt(cancelled)
        except Exception:
            # Never kill ongoing work because Windows couldn't display a dialog.
            logging.exception('Could not display Chep close dialog. Use Close Chep or the Stop Chep shortcut.')
            return
        with self.lock:
            if cancelled.is_set() or self.clients or generation != self.generation or self.stopping:
                return
            self.choose(choice)

    def choose(self, choice):
        if choice not in ('quit', 'background'):
            raise ValueError('Choose Quit completely or Run in background.')
        with self.lock:
            self._invalidate()
            if self.stopping:
                return
            self.background = choice == 'background'
            if choice == 'quit':
                self.stopping = True
                # Non-daemon: complete SQLite/worker cleanup before process exit.
                threading.Thread(target=self.shutdown, name='chep-shutdown', daemon=False).start()

    def close(self):
        with self.lock:
            self.stopping = True
            self._invalidate()
