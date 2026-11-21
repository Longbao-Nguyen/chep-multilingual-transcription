import type { Job, JobConfig, Profile } from '../types';
import { formatErrorText, parseJson } from '../utils/format';

export function JobMetadata({ job }: { job: Job }) {
  const config = parseJson<JobConfig>(job.config);
  const profile = parseJson<Profile>(job.profile);
  if (!config) return null;

  const deviceSummary =
    config.device_preference === 'kaggle'
      ? 'Online GPU (Kaggle)'
      : profile?.device === 'cuda'
        ? 'Local GPU'
        : profile?.device === 'cpu'
          ? 'CPU'
          : config.device_preference === 'cpu'
            ? 'CPU only'
            : 'Local GPU';

  return (
    <details className="job-metadata">
      <summary>
        {config.model_config.label} · {deviceSummary}
      </summary>
      <p>{profile?.model_id || config.model_config.model}</p>
      <p>
        {config.model_config.backend} · {profile?.chunk_seconds || config.model_config.chunk_seconds}s chunks
      </p>
      {profile ? (
        <p>
          {profile.compute_type} · {profile.concurrency} concurrent chunk
          {profile.concurrency !== 1 ? 's' : ''}
          {profile.device_name ? ` · ${profile.device_name}` : ''}
        </p>
      ) : (
        <p>Runtime configuration has not been confirmed.</p>
      )}
      {profile?.reason && <p>{formatErrorText(profile.reason)}</p>}
      {job.attempts && <p>Fallback attempts: {parseJson<unknown[]>(job.attempts)?.length || 0}</p>}
      {config.legacy && (
        <p>Imported from the French-only version. Unrecorded device details are unavailable.</p>
      )}
    </details>
  );
}
