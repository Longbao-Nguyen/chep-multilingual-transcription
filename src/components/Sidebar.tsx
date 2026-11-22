import { useState } from 'react';
import { ArrowDownToLine, ArrowRight, AudioLines, Cpu, FolderOpen, HardDrive, History, ShieldCheck } from 'lucide-react';
import type { AppTab, Status } from '../types';
import { formatBytes } from '../utils/format';

interface SidebarProps {
  status: Status | null;
  tab: AppTab;
  queued: number;
  completed: number;
  busy: boolean;
  locked: boolean;
  kaggleWorking: boolean;
  colabWorking: boolean;
  onSelectTab: (tab: AppTab) => void;
  onAction: (path: string, method?: string, body?: object) => void;
  onError: (message: string) => void;
}

export function Sidebar({
  status,
  tab,
  queued,
  completed,
  busy,
  locked,
  kaggleWorking,
  colabWorking,
  onSelectTab,
  onAction,
  onError,
}: SidebarProps) {
  const [colabCode, setColabCode] = useState('');
  const isColab = status?.settings.device === 'colab';

  return (
    <aside className="sidebar">
      <a
        className="brand"
        href="#"
        onClick={e => {
          e.preventDefault();
          onSelectTab('workspace');
        }}
      >
        <span className="brand-mark">
          <AudioLines size={25} />
        </span>
        <span>
          <strong>Chep</strong>
        </span>
        <small className="brand-version">v{status?.version || '0.7.0'}</small>
      </a>
      <nav aria-label="Main navigation">
        <button className={tab === 'workspace' ? 'selected' : ''} onClick={() => onSelectTab('workspace')}>
          <AudioLines size={19} />
          Transcribe<span>{queued}</span>
        </button>
        <button className={tab === 'history' ? 'selected' : ''} onClick={() => onSelectTab('history')}>
          <History size={19} />
          History<span>{completed}</span>
        </button>
      </nav>
      {!isColab ? (
        <section className="storage-card kaggle-account" aria-label="Kaggle CLI Connection">
          <div className="card-kicker">
            <ShieldCheck size={17} />
            KAGGLE CONNECTION
          </div>
          <strong role="status">
            {status?.kaggle.status === 'ready' && status.kaggle.username ? (
              <>
                Ready as <span>{status.kaggle.username}</span>
              </>
            ) : (
              <>
                Not ready <small>— authentication required</small>
              </>
            )}
          </strong>
          <button
            className="text-button"
            disabled={!status || locked || kaggleWorking || status.kaggle.status === 'ready'}
            onClick={() => {
              if (!status?.kaggle.installed) {
                onError('Download Kaggle CLI below, then choose Authenticate Kaggle CLI.');
              } else {
                onAction('/kaggle/login');
              }
            }}
          >
            <ArrowRight size={16} />
            Authenticate Kaggle CLI
          </button>
          {status?.kaggle.status === 'ready' && (
            <>
              <button
                className="text-button"
                disabled={busy || ['checking', 'ready'].includes(status.kaggle_gpu.status)}
                onClick={() => onAction('/kaggle/gpu/check')}
              >
                <Cpu size={16} />
                Check GPU condition
              </button>
              {status.kaggle_gpu.status === 'checking' && <p role="status">{status.kaggle_gpu.message}</p>}
              {status.kaggle_gpu.status === 'ready' && status.kaggle_gpu.quota && (
                <div className="kaggle-quota" role="status">
                  <span>YOUR KAGGLE QUOTA</span>
                  <strong>
                    GPU: {status.kaggle_gpu.quota.remaining} available of {status.kaggle_gpu.quota.total}
                  </strong>
                </div>
              )}
              {status.kaggle_gpu.status === 'verification_required' && (
                <p role="alert">
                  {status.kaggle_gpu.message}
                  <a href="https://www.kaggle.com/settings" target="_blank" rel="noreferrer">
                    Kaggle Setting
                  </a>
                  .
                </p>
              )}
              {['error', 'quota_exhausted', 'cleanup_pending'].includes(status.kaggle_gpu.status) && (
                <p role="alert">{status.kaggle_gpu.message}</p>
              )}
            </>
          )}
          {!status?.kaggle.installed && (
            <button
              className="text-button"
              disabled={!status || busy || kaggleWorking}
              onClick={() => onAction('/kaggle/download')}
            >
              <ArrowDownToLine size={15} />
              Download Kaggle CLI ({formatBytes(status?.kaggle.download_bytes || 0)})
            </button>
          )}
          {kaggleWorking && <p role="status">{status?.kaggle.message}</p>}
          {status?.kaggle.error && (
            <p className="job-error" role="alert">
              {status.kaggle.error}
            </p>
          )}
        </section>
      ) : (
        <section className="storage-card kaggle-account colab-account" aria-label="Colab CLI Connection">
          <div className="card-kicker">
            <ShieldCheck size={17} />
            COLAB CONNECTION
          </div>
          <strong role="status">
            {status?.colab.status === 'ready' && status.colab.username ? (
              <>
                Ready as <span>{status.colab.username}</span>
              </>
            ) : (
              <>
                Not ready <small>— authentication required</small>
              </>
            )}
          </strong>
          <button
            className="text-button"
            disabled={!status || locked || colabWorking || status.colab.status === 'ready' || status.colab.status === 'awaiting_code'}
            onClick={() => {
              if (!status?.colab.installed) {
                onError('Download Colab CLI below, then choose Authenticate Colab CLI.');
              } else {
                onAction('/colab/login');
              }
            }}
          >
            <ArrowRight size={16} />
            Authenticate Colab CLI
          </button>
          {status?.colab.status === 'awaiting_code' && (
            <form
              className="colab-code-form"
              onSubmit={e => {
                e.preventDefault();
                const value = colabCode.trim();
                if (!value) return;
                onAction('/colab/code', 'POST', { code: value });
                setColabCode('');
              }}
            >
              {status.colab.auth_url && (
                <a className="colab-code-link" href={status.colab.auth_url} target="_blank" rel="noreferrer">
                  Open Google sign-in page
                </a>
              )}
              <input
                type="text"
                aria-label="Google authorization code"
                placeholder="Paste authorization code"
                value={colabCode}
                onChange={e => setColabCode(e.target.value)}
              />
              <div className="colab-actions">
                <button type="submit" className="text-button" disabled={!colabCode.trim()}>
                  Submit code
                </button>
                <button type="button" className="text-button" onClick={() => onAction('/colab/cancel')}>
                  Cancel
                </button>
              </div>
            </form>
          )}
          {status?.colab.status === 'ready' && (
            <>
              <button
                className="text-button"
                disabled={busy || ['checking', 'ready'].includes(status.colab_gpu.status)}
                onClick={() => onAction('/colab/gpu/check')}
              >
                <Cpu size={16} />
                Check GPU condition
              </button>
              {status.colab_gpu.status === 'checking' && <p role="status">{status.colab_gpu.message}</p>}
              {status.colab_gpu.status === 'ready' && status.colab_gpu.message && (
                <p role="status">{status.colab_gpu.message}</p>
              )}
              {['error', 'quota_exhausted', 'cleanup_pending'].includes(status.colab_gpu.status) && (
                <p role="alert">{status.colab_gpu.message}</p>
              )}
              <button
                className="text-button"
                disabled={!status || locked}
                onClick={() => onAction('/colab/logout')}
              >
                Disconnect Colab
              </button>
            </>
          )}
          {!status?.colab.installed && (
            <button
              className="text-button"
              disabled={!status || busy || colabWorking}
              onClick={() => onAction('/colab/download')}
            >
              <ArrowDownToLine size={15} />
              Download Colab CLI ({formatBytes(status?.colab.download_bytes || 0)})
            </button>
          )}
          {(colabWorking || status?.colab.status === 'awaiting_code') && status?.colab.message && (
            <p role="status">{status.colab.message}</p>
          )}
          {status?.colab.status === 'downloading' && (
            <progress max={1} value={status.colab.progress} aria-label="Colab CLI download progress" />
          )}
          {status?.colab.error && (
            <p className="job-error" role="alert">
              {status.colab.error}
            </p>
          )}
        </section>
      )}
      <section className="storage-card sidebar-storage">
        <div className="card-kicker">
          <HardDrive size={17} />
          LOCAL STORAGE
        </div>
        <strong>
          {status ? formatBytes(status.free_bytes) : '--'} <small>available</small>
        </strong>
        <p className="folder-path" title={status?.data_dir}>
          {status?.data_dir || 'Reading data folder...'}
        </p>
        <button className="text-button" onClick={() => onAction('/storage/open')} disabled={!status}>
          <FolderOpen size={15} />
          Open data folder
        </button>
      </section>
    </aside>
  );
}
