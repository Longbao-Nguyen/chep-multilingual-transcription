import { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { fetchTranscript } from '../services/api';
import type { Job } from '../types';

export function TranscriptDialog({ job, onClose }: { job: Job; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [text, setText] = useState('Loading transcript...');

  useEffect(() => {
    dialog.current?.showModal();
    const abort = new AbortController();
    fetchTranscript(job.id, 'txt', abort.signal)
      .then(r => r.text())
      .then(setText)
      .catch(e => {
        if (!abort.signal.aborted) setText(String(e));
      });
    return () => abort.abort();
  }, [job.id]);

  return (
    <dialog ref={dialog} className="transcript-dialog" onCancel={onClose} aria-labelledby="transcript-heading">
      <header>
        <div>
          <span className="eyebrow">TRANSCRIPT</span>
          <h2 id="transcript-heading">{job.name}</h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close transcript">
          <X size={21} />
        </button>
      </header>
      <pre>{text}</pre>
    </dialog>
  );
}
