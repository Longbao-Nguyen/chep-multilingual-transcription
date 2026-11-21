import { History } from 'lucide-react';
import type { Job } from '../types';
import { JobItem } from './JobItem';

interface HistoryPanelProps {
  jobs: Job[];
  busy: boolean;
  pausing: boolean;
  activeJobId: string | null | undefined;
  onAction: (path: string, method?: string, body?: object) => void;
  onPreview: (job: Job) => void;
  onDownload: (job: Job, format: string) => void;
}

export function HistoryPanel({
  jobs,
  busy,
  pausing,
  activeJobId,
  onAction,
  onPreview,
  onDownload,
}: HistoryPanelProps) {
  return (
    <section className="queue-card">
      <div className="section-heading">
        <h2>
          History <span className="count">{jobs.length}</span>
        </h2>
        <span className="muted">Saved until you delete it</span>
      </div>
      {jobs.length ? (
        jobs.map(job => (
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
        ))
      ) : (
        <div className="empty-state">
          <History size={30} />
          <h3>No transcripts yet</h3>
          <p>Completed transcripts will appear here.</p>
        </div>
      )}
    </section>
  );
}
