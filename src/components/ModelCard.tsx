import { ArrowDownToLine, AudioLines, LoaderCircle, RefreshCw } from 'lucide-react';
import type { JobConfig, ModelConfig, Profile, Status } from '../types';
import { formatBytes } from '../utils/format';

interface ModelCardProps {
  status: Status | null;
  activeConfig: JobConfig | undefined;
  shownModel: ModelConfig | undefined;
  profile: Profile | undefined;
  online: boolean;
  checking: boolean;
  modelReady: boolean;
  modelWorking: boolean;
  locked: boolean;
  busy: boolean;
  running: boolean;
  pausing: boolean;
  onAction: (path: string, method?: string, body?: object) => void;
}

export function ModelCard({
  status,
  activeConfig,
  shownModel,
  profile,
  online,
  checking,
  modelReady,
  modelWorking,
  locked,
  busy,
  running,
  pausing,
  onAction,
}: ModelCardProps) {
  return (
    <section className="info-card model-card">
      <div className="card-kicker">
        <AudioLines size={17} />
        TRANSCRIPTION MODEL
      </div>
      <label className="field-label" htmlFor="language">
        Language / Transcription model
      </label>
      <select
        id="language"
        value={activeConfig?.language || status?.settings.language || 'fr'}
        disabled={locked}
        onChange={e => onAction('/settings', 'PUT', { language: e.target.value })}
      >
        {status?.languages.map(model => (
          <option key={model.language} value={model.language}>
            {model.label}
          </option>
        ))}
      </select>
      <h3>{shownModel?.label || 'Selected'} transcription</h3>
      <p className="model-id">{shownModel?.model || 'Loading model catalog...'}</p>
      {checking ? (
        <p className="configuration-status" role="status">
          <LoaderCircle size={14} className="spin" />
          Checking configuration...
        </p>
      ) : profile ? (
        <div className="engine-stats">
          <div>
            <span>Compute type</span>
            <strong>{profile.compute_type}</strong>
          </div>
          <div>
            <span>Concurrency</span>
            <strong>
              {profile.concurrency}
              <small> chunks</small>
            </strong>
          </div>
        </div>
      ) : null}
      {shownModel?.benchmark_status === 'pending' && (
        <p className="benchmark-note">No confirmed optimal configuration yet.</p>
      )}
      {!modelReady && <p role="status">{status?.model.message || 'Checking model files...'}</p>}
      {modelWorking && (
        <progress max={1} value={status?.model.progress || 0} aria-label="Model download progress" />
      )}
      {!online && status?.model.error && <p className="job-error">{status.model.error}</p>}
      {!modelReady && (
        <button
          className="small-button wide"
          disabled={modelWorking || busy || running || pausing || !status}
          onClick={() => onAction('/model/download')}
        >
          {modelWorking ? <LoaderCircle className="spin" size={15} /> : <ArrowDownToLine size={15} />}
          Download / repair model
        </button>
      )}
      {!modelReady && status && !modelWorking && (
        <p className="download-size">Download: {formatBytes(status.model.download_bytes)}</p>
      )}
      {!online && status?.runtime.status === 'error' && (
        <>
          <details className="job-metadata">
            <summary>Configuration errors</summary>
            {status.runtime.attempts.map((a, i) => (
              <p key={i}>{a.error}</p>
            ))}
          </details>
          <button className="small-button wide" disabled={locked} onClick={() => onAction('/runtime/recheck')}>
            <RefreshCw size={14} />
            Check again
          </button>
        </>
      )}
    </section>
  );
}
