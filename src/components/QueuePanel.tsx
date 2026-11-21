import { FileAudio, LoaderCircle, Pause, Play } from 'lucide-react';
import type { Job } from '../types';
import { JobItem } from './JobItem';

interface QueuePanelProps {
  jobs: Job[];
  running: boolean;
  pausing: boolean;
  busy: boolean;
  queued: number;
  checking: boolean;
  activeJobId: string | null | undefined;
  onAction: (path: string, method?: string, body?: object) => void;
  onPreview: (job: Job) => void;
  onDownload: (job: Job, format: string) => void;
}

export function QueuePanel({
  jobs,
  running,
  pausing,
  busy,
  queued,
  checking,
  activeJobId,
  onAction,
  onPreview,
  onDownload,
}: QueuePanelProps) {
  return (
    <section className="queue-card">
      <div className="section-heading">
        <div>
          <h2>
            Queue <span className="count">{jobs.length}</span>
          </h2>
          <p>
            {running
              ? 'Processing one file at a time.'
              : pausing
                ? 'Pausing the queue...'
                : 'Each file keeps the language and device selected when it was added.'}
          </p>
        </div>
        {running ? (
          <button className="small-button" disabled={busy} onClick={() => onAction('/queue/pause')}>
            <Pause size={15} />
            Pause after this file
          </button>
        ) : (
          <button
            className="primary-button"
            disabled={busy || pausing || !queued || checking}
            onClick={() => onAction('/queue/start')}
          >
            {busy || pausing ? <LoaderCircle className="spin" size={16} /> : <Play size={15} />}
            Start
          </button>
        )}
      </div>
      {jobs.length ? (
        <div className="job-list">
          {jobs.map(job => (
            <JobItem
              key={job.id}
              job={job}
              busy={busy}
              pausing={pausing}
              activeJobId={activeJobId}
              onAction={onAction}
              onPreview={onPreview}
              onDownload={onDownload}
            />
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <FileAudio size={29} />
          <h3>No files in the queue</h3>
          <p>Add a recording to start transcribing.</p>
        </div>
      )}
    </section>
  );
}
