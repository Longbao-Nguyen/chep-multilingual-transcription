import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowRight, CheckCircle2, ChevronRight, LoaderCircle, X } from 'lucide-react';
import { CleanupNotice } from './components/CleanupNotice';
import { DeviceCard } from './components/DeviceCard';
import { ExitDialog } from './components/ExitDialog';
import { GoodbyeScreen } from './components/GoodbyeScreen';
import { HistoryPanel } from './components/HistoryPanel';
import { ModelCard } from './components/ModelCard';
import { OptionalComponentsCard } from './components/OptionalComponentsCard';
import { QueuePanel } from './components/QueuePanel';
import { Sidebar } from './components/Sidebar';
import { TranscriptDialog } from './components/TranscriptDialog';
import { UploadDropzone } from './components/UploadDropzone';
import { UserGuideDrawer } from './components/UserGuideDrawer';
import { useAppStatus } from './hooks/useAppStatus';
import { useServerStopped, useUiSession } from './hooks/useUiSession';
import {
  apiRequest,
  downloadTranscriptFile,
  postLifecycleChoice,
  uploadJobFile,
} from './services/api';
import type { AppTab, ExitChoice, Job, JobConfig, Profile, UploadState } from './types';
import { parseJson } from './utils/format';

export default function App() {
  const [guideOpen, setGuideOpen] = useState(false);
  const [exitOpen, setExitOpen] = useState(false);
  const [departure, setDeparture] = useState<ExitChoice | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [dismissedCleanup, setDismissedCleanup] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [selectionPending, setSelectionPending] = useState(false);
  const [upload, setUpload] = useState<UploadState | null>(null);
  const [dragging, setDragging] = useState(false);
  const [tab, setTab] = useState<AppTab>('workspace');
  const [preview, setPreview] = useState<Job | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const uploadLock = useRef(false);

  const { status, connectionError, refresh, invalidateEpoch } = useAppStatus(departure);
  const sessionConnected = useUiSession(departure);
  const serverStopped = useServerStopped(departure);

  const dismissCleanup = useCallback((id: string) => {
    setDismissedCleanup(current => (current.includes(id) ? current : [...current, id]));
  }, []);

  const pendingCleanupIds = status?.queue.cleanup_pending.map(job => job.id).join('|') || '';
  useEffect(() => {
    const active = new Set(pendingCleanupIds ? pendingCleanupIds.split('|') : []);
    setDismissedCleanup(current => {
      const kept = current.filter(id => active.has(id));
      return kept.length === current.length ? current : kept;
    });
  }, [pendingCleanupIds]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 5000);
    return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (departure) setNotice('');
  }, [departure]);

  const chooseExit = async (choice: ExitChoice) => {
    setBusy(true);
    setError('');
    try {
      await postLifecycleChoice(choice);
      setExitOpen(false);
      setDeparture(choice);
    } catch (e) {
      setExitOpen(false);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const action = async (path: string, method = 'POST', body?: object) => {
    setBusy(true);
    setError('');
    setNotice('');
    const selectionChange = path === '/settings';
    if (selectionChange) {
      invalidateEpoch();
      setSelectionPending(true);
    }
    try {
      const response = await apiRequest(path, {
        method,
        ...(body ? { body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } } : {}),
      });
      if (method === 'DELETE' || path.endsWith('/cleanup')) {
        const result = await response.json();
        setNotice(
          result.cleanup_pending
            ? 'Job removed from the queue. Chep will retry resource cleanup automatically.'
            : method === 'DELETE'
              ? 'Job removal and cleanup completed.'
              : 'Remote cleanup completed.',
        );
      }
      // Invalidate status requests that began during the mutation too.
      invalidateEpoch();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setSelectionPending(false);
    }
  };

  const addFiles = async (files: File[]) => {
    if (uploadLock.current || selectionPending || !status || !files.length) return;
    uploadLock.current = true;
    setError('');
    try {
      for (const [index, file] of files.entries()) {
        setUpload({ name: file.name, progress: 0, index: index + 1, total: files.length });
        await uploadJobFile(file, progress => {
          setUpload({ name: file.name, progress, index: index + 1, total: files.length });
        });
        await refresh();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUpload(null);
      uploadLock.current = false;
      if (input.current) input.current.value = '';
    }
  };

  const download = async (job: Job, format: string) => {
    try {
      await downloadTranscriptFile(job, format as 'txt' | 'srt');
    } catch (e) {
      setError(String(e));
    }
  };

  const jobs = status?.queue.jobs || [];
  const currentJobs = jobs.filter(job => !['COMPLETED', 'CANCELLED'].includes(job.status));
  const pastJobs = jobs.filter(job => ['COMPLETED', 'CANCELLED'].includes(job.status)).reverse();
  const queued = jobs.filter(job => job.status === 'QUEUED').length;
  const completed = jobs.filter(job => job.status === 'COMPLETED').length;
  const running = status?.queue.state === 'RUNNING';
  const pausing = status?.queue.state === 'PAUSING';
  const activeJob = jobs.find(job => job.id === status?.queue.active_id);
  const activeConfig = parseJson<JobConfig>(activeJob?.config);
  const selectedModel = status?.languages.find(m => m.language === status.settings.language);
  const shownModel = activeConfig?.model_config || selectedModel;
  const profile = selectionPending
    ? undefined
    : activeJob
      ? parseJson<Profile>(activeJob.profile)
      : status?.runtime.status === 'ready'
        ? status.runtime.profile ?? undefined
        : undefined;
  const online = status?.settings.device === 'kaggle' || status?.settings.device === 'colab';
  const kaggleWorking = ['downloading', 'connecting'].includes(status?.kaggle.status || '');
  const colabWorking = ['downloading', 'connecting', 'awaiting_code'].includes(status?.colab.status || '');
  const modelReady = online || status?.model.status === 'ready';
  const modelWorking = !online && ['downloading', 'verifying'].includes(status?.model.status || '');
  const checking =
    !online &&
    (selectionPending ||
      status?.runtime.status === 'checking' ||
      modelWorking ||
      Boolean(activeJob && !profile));
  const locked = busy || running || pausing || Boolean(upload) || !status;

  if (departure) {
    return (
      <GoodbyeScreen
        departure={departure}
        serverStopped={serverStopped}
        onReturn={() => {
          setDeparture(null);
          void refresh();
        }}
      />
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        status={status}
        tab={tab}
        queued={queued}
        completed={completed}
        busy={busy}
        locked={locked}
        kaggleWorking={kaggleWorking}
        colabWorking={colabWorking}
        onSelectTab={setTab}
        onAction={(path, method, body) => void action(path, method, body)}
        onError={setError}
      />
      <div className="main-shell">
        <header className="topbar">
          <div>
            <span>Workspace</span>
            <ChevronRight size={14} />
            <strong>{tab === 'workspace' ? 'Transcribe' : 'History'}</strong>
          </div>
          <div className="topbar-actions">
            <button
              className="small-button guide-trigger"
              aria-haspopup="dialog"
              aria-expanded={guideOpen}
              aria-controls={guideOpen ? 'user-guide' : undefined}
              onClick={() => setGuideOpen(true)}
            >
              User Guide
            </button>
            <button
              className="small-button topbar-close"
              onClick={() => setExitOpen(true)}
              disabled={busy || Boolean(upload)}
            >
              <X size={16} />
              Close Chep
            </button>
          </div>
        </header>
        <main className="content">
          {(error || connectionError) && (
            <div className="error-banner" role="alert">
              {error || connectionError}
              <button className="icon-button" onClick={() => setError('')} aria-label="Dismiss message">
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="success-banner" role="status">
              {notice}
              <button className="icon-button" onClick={() => setNotice('')} aria-label="Dismiss message">
                <X size={16} />
              </button>
            </div>
          )}
          {status?.queue.cleanup_pending
            .filter(job => !dismissedCleanup.includes(job.id))
            .map(job => (
              <CleanupNotice key={job.id} job={job} onExpire={dismissCleanup} />
            ))}
          {!sessionConnected && status && (
            <p className="session-warning" role="status">
              Close detection is reconnecting. Use Close Chep to choose whether to quit or run in background.
            </p>
          )}
          {!status && !connectionError && (
            <div className="loading-state">
              <LoaderCircle className="spin" />
              Connecting to the application...
            </div>
          )}
          {tab === 'workspace' && (
            <div className="workspace-grid">
              <div className="primary-column">
                <UploadDropzone
                  inputRef={input}
                  dragging={dragging}
                  upload={upload}
                  selectionPending={selectionPending}
                  status={status}
                  online={online}
                  onDraggingChange={setDragging}
                  onFilesSelected={files => void addFiles(files)}
                />
                <QueuePanel
                  jobs={currentJobs}
                  running={running}
                  pausing={pausing}
                  busy={busy}
                  queued={queued}
                  checking={checking}
                  activeJobId={status?.queue.active_id}
                  onAction={(path, method, body) => void action(path, method, body)}
                  onPreview={setPreview}
                  onDownload={(job, format) => void download(job, format)}
                />
                {completed > 0 && (
                  <button className="history-link" onClick={() => setTab('history')}>
                    <CheckCircle2 size={18} />
                    {completed} completed transcript{completed !== 1 ? 's' : ''}
                    <span>
                      View history <ArrowRight size={16} />
                    </span>
                  </button>
                )}
              </div>
              <aside className="details-column">
                <DeviceCard
                  status={status}
                  profile={profile}
                  online={online}
                  checking={checking}
                  running={running}
                  pausing={pausing}
                  locked={locked}
                  busy={busy}
                  kaggleWorking={kaggleWorking}
                  colabWorking={colabWorking}
                  onAction={(path, method, body) => void action(path, method, body)}
                />
                <ModelCard
                  status={status}
                  activeConfig={activeConfig}
                  shownModel={shownModel}
                  profile={profile}
                  online={online}
                  checking={checking}
                  modelReady={modelReady}
                  modelWorking={modelWorking}
                  locked={locked}
                  busy={busy}
                  running={running}
                  pausing={pausing}
                  onAction={(path, method, body) => void action(path, method, body)}
                />
                {status && (status.components.tasks.length > 0 || status.components.setup_error) && (
                  <OptionalComponentsCard
                    components={status.components}
                    busy={busy}
                    onAction={(path, method, body) => void action(path, method, body)}
                  />
                )}
              </aside>
            </div>
          )}
          {tab === 'history' && (
            <HistoryPanel
              jobs={pastJobs}
              busy={busy}
              pausing={pausing}
              activeJobId={status?.queue.active_id}
              onAction={(path, method, body) => void action(path, method, body)}
              onPreview={setPreview}
              onDownload={(job, format) => void download(job, format)}
            />
          )}
        </main>
      </div>
      {exitOpen && (
        <ExitDialog
          busy={busy}
          onClose={() => setExitOpen(false)}
          onChoose={choice => void chooseExit(choice)}
        />
      )}
      {preview && <TranscriptDialog job={preview} onClose={() => setPreview(null)} />}
      {guideOpen && <UserGuideDrawer onClose={() => setGuideOpen(false)} />}
    </div>
  );
}
