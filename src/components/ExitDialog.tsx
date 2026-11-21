import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import type { ExitChoice } from '../types';

export function ExitDialog({
  onChoose,
  onClose,
  busy,
}: {
  onChoose: (choice: ExitChoice) => void;
  onClose: () => void;
  busy: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    dialog.current?.showModal();
  }, []);

  return (
    <dialog
      ref={dialog}
      className="exit-dialog"
      onCancel={e => {
        e.preventDefault();
        if (!busy) onClose();
      }}
      aria-labelledby="exit-heading"
    >
      <header>
        <h2 id="exit-heading">Quit Chep or keep it running?</h2>
        <button className="icon-button" aria-label="Keep using Chep" disabled={busy} onClick={onClose}>
          <X size={18} />
        </button>
      </header>
      <p>Your queue and completed transcripts are saved with either choice.</p>
      <button className="exit-choice" disabled={busy} onClick={() => onChoose('quit')}>
        <strong>Quit completely</strong>
        <span>
          Stop local work and release memory, GPU resources and application files. Choose this when you are done, or before updating or uninstalling Chep. Unfinished local work can be retried later.
        </span>
      </button>
      <button className="exit-choice" disabled={busy} onClick={() => onChoose('background')}>
        <strong>Run in background</strong>
        <span>
          Keep transcription and downloads running, and retrieve Kaggle results while the browser is closed. Reopen Chep from its shortcut whenever you need it.
        </span>
      </button>
      <p className="exit-notice">
        Kaggle notebooks already submitted can keep running online after Chep quits. Reopen Chep to retrieve their results. To stop a remote job, cancel it in the queue first.
      </p>
    </dialog>
  );
}
