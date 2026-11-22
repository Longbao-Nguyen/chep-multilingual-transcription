import { ArrowDownToLine, Cpu } from 'lucide-react';
import type { Profile, Status } from '../types';
import { formatBytes } from '../utils/format';

interface DeviceCardProps {
  status: Status | null;
  profile: Profile | undefined;
  online: boolean;
  checking: boolean;
  running: boolean;
  pausing: boolean;
  locked: boolean;
  busy: boolean;
  kaggleWorking: boolean;
  colabWorking: boolean;
  onAction: (path: string, method?: string, body?: object) => void;
}

export function DeviceCard({
  status,
  profile,
  online,
  checking,
  running,
  pausing,
  locked,
  busy,
  kaggleWorking,
  colabWorking,
  onAction,
}: DeviceCardProps) {
  const gpuDownloading = status?.components.tasks.some(
    t => t.id === 'gpu:auto' && ['queued', 'downloading', 'verifying', 'installing'].includes(t.status),
  );
  const device = status?.settings.device || 'auto';
  const isColab = device === 'colab';

  return (
    <section className="engine-card">
      <div className="card-kicker">
        <Cpu size={17} />
        PROCESSING DEVICE
      </div>
      <label className="field-label" htmlFor="device">
        Processing device
      </label>
      <select
        id="device"
        value={device}
        disabled={locked}
        onChange={e => onAction('/settings', 'PUT', { device: e.target.value })}
      >
        <option value="kaggle">Online GPU (Kaggle, recommended)</option>
        <option value="colab">Online GPU (Google Colab)</option>
        <option value="auto">Local GPU (if CUDA available)</option>
        <option value="cpu">CPU only</option>
      </select>
      {online && isColab ? (
        <div className="online-device">
          <h2>Online GPU (Colab)</h2>
          <p className="engine-reason">
            Audio is uploaded to a temporary Google Colab T4 session in your account, deleted when the queue finishes, and saved to History.
          </p>
          {!status?.colab.installed ? (
            <>
              <button
                className="small-button wide"
                disabled={busy || colabWorking}
                onClick={() => onAction('/colab/download')}
              >
                <ArrowDownToLine size={15} />
                {colabWorking ? 'Downloading...' : 'Download Colab CLI'}
              </button>
              <p className="download-size">Download: {formatBytes(status?.colab.download_bytes || 0)}</p>
            </>
          ) : null}
          {status?.colab.status === 'downloading' && (
            <progress max={1} value={status.colab.progress} aria-label="Colab CLI download progress" />
          )}
          {status?.colab.installed && status.colab.error && (
            <button
              className="text-button"
              disabled={busy || colabWorking}
              onClick={() => onAction('/colab/download')}
            >
              Repair Colab CLI
            </button>
          )}
        </div>
      ) : online ? (
        <div className="online-device">
          <h2>Online GPU</h2>
          <p className="engine-reason">
            Audio is processed on your private Kaggle account, deleted after completion, and saved to History. Kaggle includes 30 free GPU hours/week.
          </p>
          {!status?.kaggle.installed ? (
            <>
              <button
                className="small-button wide"
                disabled={busy || kaggleWorking}
                onClick={() => onAction('/kaggle/download')}
              >
                <ArrowDownToLine size={15} />
                {kaggleWorking ? 'Downloading...' : 'Download Kaggle CLI'}
              </button>
              <p className="download-size">Download: {formatBytes(status?.kaggle.download_bytes || 0)}</p>
            </>
          ) : null}
          {status?.kaggle.status === 'downloading' && (
            <progress max={1} value={status.kaggle.progress} aria-label="Kaggle CLI download progress" />
          )}
          {status?.kaggle.installed && status.kaggle.error && (
            <button
              className="text-button"
              disabled={busy || kaggleWorking}
              onClick={() => onAction('/kaggle/download')}
            >
              Repair Kaggle CLI
            </button>
          )}
        </div>
      ) : (
        <>
          <h2>{profile ? (profile.device !== 'cpu' ? 'GPU' : 'CPU') : checking ? 'Checking device' : 'Awaiting model'}</h2>
          {status?.settings.device === 'cpu' && status.components.gpu.compatible && (
            <p className="engine-reason">
              CPU processing may be significantly slower than GPU. For faster transcription, we recommend using a compatible GPU.
            </p>
          )}
          {status?.settings.device === 'auto' && (
            <>
              <p className="engine-reason" role="status">
                {status.components.gpu.reason}
              </p>
              {status.components.gpu.compatible && status.components.gpu.status === 'missing' && (
                <>
                  <button
                    className="small-button wide"
                    disabled={busy || gpuDownloading}
                    onClick={() => {
                      if (confirm(`Download GPU Runtime (${formatBytes(status.components.gpu.download_bytes)})?`)) {
                        onAction('/components/download', 'POST', { action: 'download', component: 'gpu:auto' });
                      }
                    }}
                  >
                    <ArrowDownToLine size={15} />
                    Download GPU Runtime
                  </button>
                  <p className="download-size">Download: {formatBytes(status.components.gpu.download_bytes)}</p>
                </>
              )}
              <button className="text-button" disabled={busy} onClick={() => onAction('/components/recheck')}>
                Check GPU compatibility again
              </button>
            </>
          )}
        </>
      )}
      {(running || pausing) && (
        <p className="engine-reason">Language and device are locked while the queue is running.</p>
      )}
    </section>
  );
}
