import { ArrowDownToLine, Check, FileAudio, LoaderCircle, RefreshCw, Trash2, X } from 'lucide-react';
import type { Job, Remote } from '../types';
import {
  ACTIVE_JOB_STATUSES,
  formatBytes,
  formatDuration,
  formatErrorText,
  jobStatusLabel,
  parseJson,
} from '../utils/format';
import { JobMetadata } from './JobMetadata';

interface JobItemProps {
  job: Job;
  busy: boolean;
  pausing: boolean;
  activeJobId: string | null | undefined;
  onAction: (path: string, method?: string, body?: object) => void;
  onPreview: (job: Job) => void;
  onDownload: (job: Job, format: string) => void;
}

export function JobItem({
  job,
  busy,
  pausing,
  activeJobId,
  onAction,
  onPreview,
  onDownload,
}: JobItemProps) {
  const isActive = ACTIVE_JOB_STATUSES.has(job.status);
  const config = parseJson<{ device_preference?: string }>(job.config);
  const remote = job.remote ? parseJson<Remote>(job.remote) : undefined;
  const provider = remote?.provider || (config?.device_preference === 'colab' ? 'colab' : 'kaggle');
  const showRemoteCleanup =
    remote &&
    !remote.cleaned &&
    (remote.cleanup_error || ['COMPLETED', 'CANCELLED', 'FAILED'].includes(job.status));

  return (
    <article className="job">
      <div className={`file-icon ${job.status === 'COMPLETED' ? 'done' : ''}`}>
        {job.status === 'COMPLETED' ? <Check size={21} /> : <FileAudio size={21} />}
      </div>
      <div className="job-main">
        <div className="job-heading">
          <h3 title={job.name}>{job.name}</h3>
          <span className={`badge ${job.status.toLowerCase()}`}>
            {isActive && <LoaderCircle size={12} className="spin" />}
            {jobStatusLabel(job.status, config?.device_preference)}
          </span>
        </div>
        <p>
          {formatBytes(job.size)}
          {job.duration ? ` · ${formatDuration(job.duration)}` : ''}
          {isActive ? ` · ${Math.round(job.progress * 100)}%` : ''}
        </p>
        <JobMetadata job={job} />
        {showRemoteCleanup && (
          <div className="remote-session">
            {remote.cleanup_error && <p className="job-error">{remote.cleanup_error}</p>}
            {['COMPLETED', 'CANCELLED', 'FAILED'].includes(job.status) && (
              <button
                className="text-button"
                disabled={busy || activeJobId === job.id}
                onClick={() => onAction(`/jobs/${job.id}/cleanup`)}
              >
                {provider === 'colab' ? 'Release Colab runtime' : 'Remove files from Kaggle'}
              </button>
            )}
          </div>
        )}
        {isActive && <progress value={job.progress} max={1} aria-label={`Progress for ${job.name}`} />}
        {job.error && <p className="job-error">{formatErrorText(job.error)}</p>}
      </div>
      <div className="job-actions">
        {job.status === 'COMPLETED' && (
          <>
            <button className="text-button" onClick={() => onPreview(job)}>
              View
            </button>
            <button className="small-button" onClick={() => onDownload(job, 'txt')}>
              <ArrowDownToLine size={14} />
              TXT
            </button>
            <button className="small-button" onClick={() => onDownload(job, 'srt')}>
              SRT
            </button>
          </>
        )}
        {['FAILED', 'INTERRUPTED'].includes(job.status) && (
          <button className="small-button" disabled={busy} onClick={() => onAction(`/jobs/${job.id}/retry`)}>
            <RefreshCw size={14} />
            Retry
          </button>
        )}
        {job.status === 'QUEUED' && (
          <button
            className="icon-button"
            aria-label={`Remove ${job.name} from queue`}
            disabled={busy}
            onClick={() => onAction(`/jobs/${job.id}`, 'DELETE')}
          >
            <X size={18} />
          </button>
        )}
        {isActive && (
          <button
            className="icon-button"
            aria-label={`Cancel ${job.name}`}
            disabled={busy || pausing}
            onClick={() => {
              if (confirm('Cancel this file and pause the queue? Other queued files will be kept.')) {
                onAction(`/jobs/${job.id}/cancel`);
              }
            }}
          >
            <X size={18} />
          </button>
        )}
        {!isActive && job.status !== 'QUEUED' && (
          <button
            className="icon-button"
            aria-label={`Delete ${job.name}`}
            disabled={busy || activeJobId === job.id}
            onClick={() => {
              if (confirm('Delete this record and its saved transcript? Your original file will be kept.')) {
                onAction(`/jobs/${job.id}`, 'DELETE');
              }
            }}
          >
            <Trash2 size={16} />
          </button>
        )}
      </div>
    </article>
  );
}
