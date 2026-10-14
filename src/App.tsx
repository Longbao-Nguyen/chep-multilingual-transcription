import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDownToLine, ArrowRight, AudioLines, Check, CheckCircle2, ChevronRight, Cpu, FileAudio, FolderOpen, HardDrive, History, LoaderCircle, Pause, Play, Plus, RefreshCw, ShieldCheck, Trash2, Upload, X } from 'lucide-react';

type Profile = { device: string; device_name: string; compute_type: string; concurrency: number; reason: string; language?: string; model_id?: string; chunk_seconds?: number };
type ModelConfig = { label: string; model: string; ct2_model: string | null; backend: string; chunk_seconds: number; benchmark_status: string; validation_status: string };
type JobConfig = { language: string; device_preference: string; model_config: ModelConfig; legacy?: boolean };
type Job = { id: string; name: string; size: number; status: string; progress: number; duration: number | null; processed: number; error: string | null; profile: string | null; config: string | null; attempts: string | null; remote?: string | null };
type Remote = { kernel: string; dataset: string; cleaned: boolean; cleanup_error?: string };
type PendingCleanup = { id: string; name: string; owner?: string; errors: Record<string, string> };
type Status = {
  kaggle: { installed: boolean; status: string; username: string | null; message: string; progress: number; error: string | null; download_bytes: number };
  components: { gpu: { compatible: boolean; status: string; reason: string; download_bytes: number }; paused: boolean; setup_error?: string; tasks: { id: string; label: string; status: string; progress: number; message: string; error: string | null }[] };
  queue: { state: string; active_id: string | null; jobs: Job[]; cleanup_pending: PendingCleanup[] };
  kaggle_gpu: { status: string; username: string | null; message: string; quota: { used: string; remaining: string; total: string; refreshAt?: string } | null };
  model: { status: string; progress: number; message: string; error: string | null; download_bytes: number };
  runtime: { status: string; profile: Profile | null; message: string; attempts: { error: string }[] };
  languages: (ModelConfig & { language: string })[];
  hardware: { cpu_name: string; gpus: { index: number; name: string; total_mib: number; free_mib: number }[] };
  settings: { device: string; language: string }; data_dir: string; free_bytes: number; version: string;
};
const ACTIVE = new Set(['LOADING', 'PREPROCESSING', 'TRANSCRIBING', 'FINALIZING', 'UPLOADING', 'WAITING_GPU', 'REMOTE_RUNNING', 'DOWNLOADING']);
const LABELS: Record<string, string> = { UPLOADING: 'Uploading to Kaggle', WAITING_GPU: 'Waiting for Kaggle GPU', REMOTE_RUNNING: 'Processing on Kaggle', DOWNLOADING: 'Retrieving transcript', QUEUED: 'Queued', LOADING: 'Loading model', PREPROCESSING: 'Preparing audio', TRANSCRIBING: 'Transcribing', FINALIZING: 'Saving', COMPLETED: 'Completed', FAILED: 'Failed', INTERRUPTED: 'Interrupted', CANCELLED: 'Cancelled' };
const bytes = (n: number) => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(1)} GB` : `${(n / 1024 ** 2).toFixed(1)} MB`;
const duration = (n: number) => `${Math.floor(n / 60)}:${Math.floor(n % 60).toString().padStart(2, '0')}`;
function parse<T>(value: string | null | undefined): T | undefined { try { return value ? JSON.parse(value) as T : undefined; } catch { return undefined; } }
function errorText(value: string) { return /[\u0102-\u01b0\u1ea0-\u1ef9]/i.test(value) ? 'This file failed in an earlier app version. Retry it or check its original log for details.' : value; }
function initialToken() {
  const token = new URLSearchParams(location.hash.slice(1)).get('token');
  if (token) { sessionStorage.setItem('ft-token', token); history.replaceState(null, '', location.pathname); }
  return token || sessionStorage.getItem('ft-token') || '';
}
const token = initialToken();
async function request(path: string, options: RequestInit = {}) {
  const response = await fetch(`/api${path}`, { ...options, headers: { Authorization: `Bearer ${token}`, ...options.headers } });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : 'The action could not be completed.');
  }
  return response;
}
function Transcript({ job, onClose }: { job: Job; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [text, setText] = useState('Loading transcript...');
  useEffect(() => {
    dialog.current?.showModal();
    const abort = new AbortController();
    request(`/jobs/${job.id}/transcript`, { signal: abort.signal }).then(r => r.text()).then(setText).catch(e => { if (!abort.signal.aborted) setText(String(e)); });
    return () => abort.abort();
  }, [job.id]);
  return <dialog ref={dialog} className="transcript-dialog" onCancel={onClose} aria-labelledby="transcript-heading">
    <header><div><span className="eyebrow">TRANSCRIPT</span><h2 id="transcript-heading">{job.name}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close transcript"><X size={21}/></button></header>
    <pre>{text}</pre>
  </dialog>;
}
function JobMetadata({ job }: { job: Job }) {
  const config = parse<JobConfig>(job.config);
  const profile = parse<Profile>(job.profile);
  if (!config) return null;
  return <details className="job-metadata"><summary>{config.model_config.label} · {config.device_preference === 'kaggle' ? 'Online GPU (Kaggle)' : profile?.device === 'cuda' ? 'Local GPU' : profile?.device === 'cpu' ? 'CPU' : config.device_preference === 'cpu' ? 'CPU only' : 'Local GPU'}</summary>
    <p>{profile?.model_id || config.model_config.model}</p>
    <p>{config.model_config.backend} · {profile?.chunk_seconds || config.model_config.chunk_seconds}s chunks</p>
    {profile ? <p>{profile.compute_type} · {profile.concurrency} concurrent chunk{profile.concurrency !== 1 ? 's' : ''}{profile.device_name ? ` · ${profile.device_name}` : ''}</p> : <p>Runtime configuration has not been confirmed.</p>}
    {profile?.reason && <p>{errorText(profile.reason)}</p>}
    {job.attempts && <p>Fallback attempts: {parse<unknown[]>(job.attempts)?.length || 0}</p>}
    {config.legacy && <p>Imported from the French-only version. Unrecorded device details are unavailable.</p>}
  </details>;
}
function ExitDialog({ onChoose, onClose, busy }: { onChoose: (choice: 'quit' | 'background') => void; onClose: () => void; busy: boolean }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  return <dialog ref={dialog} className="exit-dialog" onCancel={e => { e.preventDefault(); if (!busy) onClose(); }} aria-labelledby="exit-heading">
    <header><h2 id="exit-heading">Quit Chep or keep it running?</h2><button className="icon-button" aria-label="Keep using Chep" disabled={busy} onClick={onClose}><X size={18}/></button></header>
    <p>Your queue and completed transcripts are saved with either choice.</p>
    <button className="exit-choice" disabled={busy} onClick={() => onChoose('quit')}><strong>Quit completely</strong><span>Stop local work and release memory, GPU resources and application files. Choose this when you are done, or before updating or uninstalling Chep. Unfinished local work can be retried later.</span></button>
    <button className="exit-choice" disabled={busy} onClick={() => onChoose('background')}><strong>Run in background</strong><span>Keep transcription and downloads running, and retrieve Kaggle results while the browser is closed. Reopen Chep from its shortcut whenever you need it.</span></button>
    <p className="exit-notice">Kaggle notebooks already submitted can keep running online after Chep quits. Reopen Chep to retrieve their results. To stop a remote job, cancel it in the queue first.</p>
  </dialog>;
}
function UserGuide({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.current?.showModal();
    return () => { document.body.style.overflow = previousOverflow; previousFocus?.focus(); };
  }, []);
  return <dialog ref={dialog} id="user-guide" className="guide-drawer" aria-labelledby="guide-title" onCancel={e => { e.preventDefault(); onClose(); }}>
    <header><h2 id="guide-title">Chep — User Guide</h2><button className="icon-button" aria-label="Close user guide" onClick={onClose} autoFocus><X size={22}/></button></header>
    <div className="guide-content">
      <section><h3>1. Common Use</h3>
        <h4>Step 1: Select a Processing Device</h4>
        <p>Each processing option has its own advantages:</p>
        <ul>
          <li><strong>Online GPU (Recommended):</strong> Kaggle typically provides up to 30 hours of free GPU usage per week. The models were benchmarked and optimized on Kaggle GPUs during development, making this the recommended option for most users.</li>
          <li><strong>Local GPU:</strong> Provides greater privacy by keeping all input audio files and output transcripts on your device. This option is available only for compatible NVIDIA GPUs with CUDA support.</li>
          <li><strong>CPU Only:</strong> Offers the same level of privacy as Local GPU but processes audio significantly more slowly (approximately 25× slower, depending on the hardware and model). Best suited for short audio files.</li>
        </ul>
        <h4>Step 2: Select a Language</h4>
        <p>Choose the language of the audio you want to transcribe. This version of Chep supports French, English, Spanish, German, and Vietnamese.</p>
        <h4>Step 3: Add Audio Files to the Queue</h4>
        <p>Select or drag and drop the audio files you want to transcribe to create a processing queue.</p>
        <ul><li>Click <strong>Start</strong> to process the files sequentially.</li><li>You can remove individual files from the queue without affecting other files.</li><li>You also can cancel an active transcription task if needed. This will pause the queue until you click <strong>Start</strong> again.</li></ul>
        <h4>Step 4: Access Your Transcripts</h4>
        <p>Once processing is complete, your transcripts will appear in the <strong>History</strong> tab, where you can view and download them.</p>
      </section>
      <section><h3>2. Setup &amp; Installation</h3>
        <h4>CPU Only</h4>
        <p>CPU processing is available by default and does not require a dedicated GPU. However, you must download the appropriate language model before running local transcription.</p>
        <h4>Local GPU</h4>
        <p>Local GPU processing requires a compatible NVIDIA GPU. You need to:</p>
        <ol><li>Download the required CUDA runtime components.</li><li>Download the language models you want to use.</li></ol>
        <p>Chep will be able to run transcription directly on your device when the required components are installed.</p>
        <h4>Online GPU (Kaggle — Recommended)</h4>
        <p>Online GPU is a convenient option that allows you to use Kaggle's free GPU resources without requiring powerful local hardware.</p>
        <ol><li>Click <strong>Download Kaggle CLI</strong>.</li><li>Create a Kaggle account or sign in to your existing account.</li><li>Click <strong>Authenticate Kaggle CLI</strong> and approve the authorization request.</li><li>Click <strong>Check GPU condition</strong> and verify your phone number to access GPU.</li></ol>
        <p>Once finished, you can start transcribing without downloading language models or CUDA components to your computer.</p>
        <p><strong>Note:</strong> Online GPU processing requires an internet connection. Your audio files will be uploaded to your private Kaggle account for processing and deleted after.</p>
      </section>
      <section><h3>3. Important Notes</h3>
        <p><strong>Always follow the steps in the correct order:</strong></p>
        <ol><li>Select your processing device.</li><li>Select your transcription language and model.</li><li>Add your audio files to the queue.</li><li>Click <strong>Start</strong> to begin transcription.</li></ol>
        <p><strong>Processing device and model selections are locked for each file once it is added to the queue.</strong> Changing these settings afterward will only affect newly added files. To use a different processing device or model for an existing file, remove that file from the queue and add it again with the desired settings.</p>
      </section>
    </div>
  </dialog>;
}
export default function App() {
  const [guideOpen, setGuideOpen] = useState(false);
  const [exitOpen, setExitOpen] = useState(false);
  const [departure, setDeparture] = useState<'quit' | 'background' | null>(null);
  const [serverStopped, setServerStopped] = useState(false);
  const [sessionConnected, setSessionConnected] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [connectionError, setConnectionError] = useState('');
  const [busy, setBusy] = useState(false);
  const [selectionPending, setSelectionPending] = useState(false);
  const [upload, setUpload] = useState<{ name: string; progress: number; index: number; total: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [tab, setTab] = useState<'workspace' | 'history'>('workspace');
  const [preview, setPreview] = useState<Job | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const uploadLock = useRef(false);
  const mounted = useRef(true);
  const epoch = useRef(0);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    const generation = epoch.current;
    const response = await request('/status', { signal });
    const next = await response.json() as Status;
    if (mounted.current && generation === epoch.current) { setStatus(next); setConnectionError(''); }
  }, []);
  useEffect(() => {
    mounted.current = true;
    if (departure) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try { await refresh(abort.signal); } catch (e) { if (!abort.signal.aborted) setConnectionError(e instanceof Error ? e.message : 'Connection lost.'); }
      if (!abort.signal.aborted) timer = setTimeout(poll, 1500);
    };
    void poll();
    return () => { mounted.current = false; abort.abort(); clearTimeout(timer); };
  }, [refresh, departure]);
  useEffect(() => {
    if (departure) return;
    let disposed = false;
    let socket: WebSocket | undefined;
    let timer: ReturnType<typeof setTimeout>;
    const connect = () => {
      socket = new WebSocket(`ws://${location.host}/api/ui-session`);
      socket.onopen = () => socket?.send(JSON.stringify({ token }));
      socket.onmessage = () => { if (!disposed) setSessionConnected(true); };
      socket.onclose = () => { if (!disposed) { setSessionConnected(false); timer = setTimeout(connect, 1500); } };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => { disposed = true; clearTimeout(timer); socket?.close(); };
  }, [departure]);
  useEffect(() => {
    if (departure !== 'quit') return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const check = async () => {
      try { await request('/status', { signal: AbortSignal.timeout(5000) }); }
      catch (e) { if (!disposed && e instanceof TypeError) { setServerStopped(true); return; } }
      if (!disposed) timer = setTimeout(check, 800);
    };
    void check();
    return () => { disposed = true; clearTimeout(timer); };
  }, [departure]);
  const chooseExit = async (choice: 'quit' | 'background') => {
    setBusy(true); setError('');
    try {
      await request('/lifecycle/choose', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ choice }) });
      setExitOpen(false); setDeparture(choice);
    } catch (e) { setExitOpen(false); setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const action = async (path: string, method = 'POST', body?: object) => {
    setBusy(true); setError(''); setNotice('');
    const selectionChange = path === '/settings';
    if (selectionChange) { epoch.current++; setSelectionPending(true); }
    try {
      const response = await request(path, { method, ...(body ? { body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } } : {}) });
      if (method === 'DELETE' || path.endsWith('/cleanup')) {
        const result = await response.json();
        setNotice(result.cleanup_pending ? 'Job removed from the queue. Resource cleanup is pending; use Retry cleanup below.' : method === 'DELETE' ? 'Job removal and cleanup completed.' : 'Kaggle cleanup completed.');
      }
      // Invalidate status requests that began during the mutation too.
      epoch.current++;
      await refresh();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); setSelectionPending(false); }
  };
  const addFiles = async (files: File[]) => {
    if (uploadLock.current || selectionPending || !status || !files.length) return;
    uploadLock.current = true; setError('');
    try {
      for (const [index, file] of files.entries()) {
        setUpload({ name: file.name, progress: 0, index: index + 1, total: files.length });
        await new Promise<void>((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhr.open('POST', `/api/jobs?name=${encodeURIComponent(file.name)}`);
          xhr.setRequestHeader('Authorization', `Bearer ${token}`);
          xhr.upload.onprogress = event => { if (event.lengthComputable) setUpload({ name: file.name, progress: event.loaded / event.total, index: index + 1, total: files.length }); };
          xhr.onerror = () => reject(new Error('Connection lost while adding the file.'));
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) resolve();
            else { let message = 'Could not add the file.'; try { message = JSON.parse(xhr.responseText).detail || message; } catch {} reject(new Error(message)); }
          };
          xhr.send(file);
        });
        await refresh();
      }
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setUpload(null); uploadLock.current = false; if (input.current) input.current.value = ''; }
  };
  const download = async (job: Job, format: string) => {
    try {
      const response = await request(`/jobs/${job.id}/transcript?format=${format}`);
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${job.name.replace(/\.[^.]+$/, '')}.${format}`;
      anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (e) { setError(String(e)); }
  };
  const jobs = status?.queue.jobs || [];
  const currentJobs = jobs.filter(job => !['COMPLETED', 'CANCELLED'].includes(job.status));
  const pastJobs = jobs.filter(job => ['COMPLETED', 'CANCELLED'].includes(job.status)).reverse();
  const queued = jobs.filter(job => job.status === 'QUEUED').length;
  const completed = jobs.filter(job => job.status === 'COMPLETED').length;
  const running = status?.queue.state === 'RUNNING';
  const pausing = status?.queue.state === 'PAUSING';
  const activeJob = jobs.find(job => job.id === status?.queue.active_id);
  const activeConfig = parse<JobConfig>(activeJob?.config);
  const selectedModel = status?.languages.find(m => m.language === status.settings.language);
  const shownModel = activeConfig?.model_config || selectedModel;
  const profile = selectionPending ? undefined : activeJob ? parse<Profile>(activeJob.profile) : status?.runtime.status === 'ready' ? status.runtime.profile : undefined;
  const online = status?.settings.device === 'kaggle';
  const kaggleWorking = ['downloading', 'connecting'].includes(status?.kaggle.status || '');
  const modelReady = online || status?.model.status === 'ready';
  const modelWorking = !online && ['downloading', 'verifying'].includes(status?.model.status || '');
  const checking = !online && (selectionPending || status?.runtime.status === 'checking' || modelWorking || Boolean(activeJob && !profile));
  const locked = busy || running || pausing || Boolean(upload) || !status;
  const renderJob = (job: Job) => <article className="job" key={job.id}>
    <div className={`file-icon ${job.status === 'COMPLETED' ? 'done' : ''}`}>{job.status === 'COMPLETED' ? <Check size={21}/> : <FileAudio size={21}/>}</div>
    <div className="job-main"><div className="job-heading"><h3 title={job.name}>{job.name}</h3><span className={`badge ${job.status.toLowerCase()}`}>{ACTIVE.has(job.status) && <LoaderCircle size={12} className="spin"/>}{LABELS[job.status] || job.status}</span></div>
      <p>{bytes(job.size)}{job.duration ? ` · ${duration(job.duration)}` : ''}{ACTIVE.has(job.status) ? ` · ${Math.round(job.progress * 100)}%` : ''}</p>
      <JobMetadata job={job}/>
      {job.remote && (() => { const remote = parse<Remote>(job.remote); return remote && !remote.cleaned ? <div className="remote-session">
        <a href={`https://www.kaggle.com/code/${remote.kernel}`} target="_blank" rel="noreferrer">Open Kaggle session</a>
        {remote.cleanup_error && <p className="job-error">{remote.cleanup_error}</p>}
        {['COMPLETED','CANCELLED','FAILED'].includes(job.status) && <button className="text-button" disabled={busy || status?.queue.active_id === job.id} onClick={() => void action(`/jobs/${job.id}/cleanup`)}>Remove files from Kaggle</button>}
      </div> : null; })()}
      {ACTIVE.has(job.status) && <progress value={job.progress} max={1} aria-label={`Progress for ${job.name}`}/>}
      {job.error && <p className="job-error">{errorText(job.error)}</p>}
    </div>
    <div className="job-actions">
      {job.status === 'COMPLETED' && <><button className="text-button" onClick={() => setPreview(job)}>View</button><button className="small-button" onClick={() => void download(job, 'txt')}><ArrowDownToLine size={14}/>TXT</button><button className="small-button" onClick={() => void download(job, 'srt')}>SRT</button></>}
      {['FAILED', 'INTERRUPTED'].includes(job.status) && <button className="small-button" disabled={busy} onClick={() => void action(`/jobs/${job.id}/retry`)}><RefreshCw size={14}/>Retry</button>}
      {job.status === 'QUEUED' && <button className="icon-button" aria-label={`Remove ${job.name} from queue`} disabled={busy} onClick={() => void action(`/jobs/${job.id}`, 'DELETE')}><X size={18}/></button>}
      {ACTIVE.has(job.status) && <button className="icon-button" aria-label={`Cancel ${job.name}`} disabled={busy || pausing} onClick={() => { if (confirm('Cancel this file and pause the queue? Other queued files will be kept.')) void action(`/jobs/${job.id}/cancel`); }}><X size={18}/></button>}
      {!ACTIVE.has(job.status) && job.status !== 'QUEUED' && <button className="icon-button" aria-label={`Delete ${job.name}`} disabled={busy || status?.queue.active_id === job.id} onClick={() => { if (confirm('Delete this record and its saved transcript? Your original file will be kept.')) void action(`/jobs/${job.id}`, 'DELETE'); }}><Trash2 size={16}/></button>}
    </div>
  </article>;
  if (departure) return <main className="goodbye">
    <h1>{departure === 'background' ? 'Chep is running in background.' : serverStopped ? 'Chep has stopped.' : 'Quitting Chep...'}</h1>
    <p>{departure === 'background' ? 'Transcription and downloads can continue. You can close this tab.' : serverStopped ? 'Application files and local resources have been released. You can close this tab.' : 'Stopping local work and saving your queue. Please wait before updating or removing the app.'}</p>
    {departure === 'background' && <button className="primary-button" onClick={() => { setDeparture(null); void refresh(); }}>Return to Chep</button>}
    {departure === 'quit' && !serverStopped && <LoaderCircle className="spin"/>}
  </main>;
  return <div className="app-shell">
    <aside className="sidebar"><a className="brand" href="#" onClick={e => { e.preventDefault(); setTab('workspace'); }}><span className="brand-mark"><AudioLines size={25}/></span><span><strong>Chep</strong></span><small className="brand-version">v{status?.version || '0.7.0'}</small></a>
      <nav aria-label="Main navigation"><button className={tab === 'workspace' ? 'selected' : ''} onClick={() => setTab('workspace')}><AudioLines size={19}/>Transcribe<span>{queued}</span></button><button className={tab === 'history' ? 'selected' : ''} onClick={() => setTab('history')}><History size={19}/>History<span>{completed}</span></button></nav>
      <section className="storage-card kaggle-account" aria-label="Kaggle CLI Connection">
        <div className="card-kicker"><ShieldCheck size={17}/>KAGGLE CONNECTION</div>
        <strong role="status">{status?.kaggle.status === 'ready' && status.kaggle.username ? <>Ready as <span>{status.kaggle.username}</span></> : <>Not ready <small>— authentication required</small></>}</strong>
        <button className="text-button" disabled={!status || locked || kaggleWorking || status.kaggle.status === 'ready'} onClick={() => {
          if (!status?.kaggle.installed) setError('Download Kaggle CLI below, then choose Authenticate Kaggle CLI.');
          else void action('/kaggle/login');
        }}><ArrowRight size={16}/>Authenticate Kaggle CLI</button>
        {status?.kaggle.status === 'ready' && <>
          <button className="text-button" disabled={busy || ['checking', 'ready'].includes(status.kaggle_gpu.status)} onClick={() => void action('/kaggle/gpu/check')}><Cpu size={16}/>Check GPU condition</button>
          {status.kaggle_gpu.status === 'checking' && <p role="status">{status.kaggle_gpu.message}</p>}
          {status.kaggle_gpu.status === 'ready' && status.kaggle_gpu.quota && <div className="kaggle-quota" role="status"><span>YOUR KAGGLE QUOTA</span><strong>GPU: {status.kaggle_gpu.quota.remaining} available of {status.kaggle_gpu.quota.total}</strong></div>}
          {status.kaggle_gpu.status === 'verification_required' && <p role="alert">{status.kaggle_gpu.message}<a href="https://www.kaggle.com/settings" target="_blank" rel="noreferrer">Kaggle Setting</a>.</p>}
          {['error', 'quota_exhausted', 'cleanup_pending'].includes(status.kaggle_gpu.status) && <p role="alert">{status.kaggle_gpu.message}</p>}
        </>}
        {!status?.kaggle.installed && <button className="text-button" disabled={!status || busy || kaggleWorking} onClick={() => void action('/kaggle/download')}><ArrowDownToLine size={15}/>Download Kaggle CLI ({bytes(status?.kaggle.download_bytes || 0)})</button>}
        {kaggleWorking && <p role="status">{status?.kaggle.message}</p>}
        {status?.kaggle.error && <p className="job-error" role="alert">{status.kaggle.error}</p>}
        <a href="https://www.kaggle.com/account/login" target="_blank" rel="noreferrer">Create/Sign in an account</a>
      </section>
      <section className="storage-card sidebar-storage"><div className="card-kicker"><HardDrive size={17}/>LOCAL STORAGE</div><strong>{status ? bytes(status.free_bytes) : '--'} <small>available</small></strong><p className="folder-path" title={status?.data_dir}>{status?.data_dir || 'Reading data folder...'}</p><button className="text-button" onClick={() => void action('/storage/open')} disabled={!status}><FolderOpen size={15}/>Open data folder</button></section>
    </aside>
    <div className="main-shell"><header className="topbar"><div><span>Workspace</span><ChevronRight size={14}/><strong>{tab === 'workspace' ? 'Transcribe' : 'History'}</strong></div><div className="topbar-actions"><button className="small-button guide-trigger" aria-haspopup="dialog" aria-expanded={guideOpen} aria-controls={guideOpen ? 'user-guide' : undefined} onClick={() => setGuideOpen(true)}>User Guide</button><button className="small-button topbar-close" onClick={() => setExitOpen(true)} disabled={busy || Boolean(upload)}><X size={16}/>Close Chep</button></div></header>
      <main className="content">
        {(error || connectionError) && <div className="error-banner" role="alert">{error || connectionError}<button className="icon-button" onClick={() => setError('')} aria-label="Dismiss message"><X size={16}/></button></div>}
        {notice && <div className="success-banner" role="status">{notice}<button className="icon-button" onClick={() => setNotice('')} aria-label="Dismiss message"><X size={16}/></button></div>}
        {status?.queue.cleanup_pending.map(job => <div className="cleanup-banner" key={job.id}><span>{job.name}: cleanup pending{job.owner ? ` in ${job.owner}'s Kaggle account` : ''}. {Object.values(job.errors).join(' ') || 'Retry to finish removing resources.'}</span><button className="small-button" disabled={busy} onClick={() => void action(`/jobs/${job.id}/cleanup`)}>Retry cleanup</button></div>)}
        {!sessionConnected && status && <p className="session-warning" role="status">Close detection is reconnecting. Use Close Chep to choose whether to quit or run in background.</p>}
        {!status && !connectionError && <div className="loading-state"><LoaderCircle className="spin"/>Connecting to the application...</div>}
        {tab === 'workspace' && <div className="workspace-grid"><div className="primary-column">
          <section className={`upload-card ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); void addFiles(Array.from(e.dataTransfer.files)); }}>
            <span className="upload-icon"><Upload size={26}/></span><h2>Add your audio</h2><p>Drag files here, or choose them from your computer.</p><button className="primary-button" disabled={Boolean(upload) || selectionPending || !status} onClick={() => input.current?.click()}><Plus size={17}/>Choose files</button><input ref={input} type="file" multiple hidden accept="audio/*,video/*,.m4a,.mkv,.wma,.opus,.flac" onChange={e => void addFiles(Array.from(e.target.files || []))}/><small>MP3, WAV, M4A, MP4 and other common formats · up to {online ? '2' : '20'} GB per file</small>
            {upload && <div className="upload-progress" role="status"><span>{upload.index}/{upload.total} · {upload.name}</span><progress max={1} value={upload.progress}/><small>Copying to local storage · {Math.round(upload.progress * 100)}%</small></div>}
          </section>
          <section className="queue-card"><div className="section-heading"><div><h2>Queue <span className="count">{currentJobs.length}</span></h2><p>{running ? 'Processing one file at a time.' : pausing ? 'Pausing the queue...' : 'Each file keeps the language and device selected when it was added.'}</p></div>{running ? <button className="small-button" disabled={busy} onClick={() => void action('/queue/pause')}><Pause size={15}/>Pause after this file</button> : <button className="primary-button" disabled={busy || pausing || !queued || checking} onClick={() => void action('/queue/start')}>{busy || pausing ? <LoaderCircle className="spin" size={16}/> : <Play size={15}/>}Start</button>}</div>
            {currentJobs.length ? <div className="job-list">{currentJobs.map(renderJob)}</div> : <div className="empty-state"><FileAudio size={29}/><h3>No files in the queue</h3><p>Add a recording to start transcribing.</p></div>}
          </section>
          {completed > 0 && <button className="history-link" onClick={() => setTab('history')}><CheckCircle2 size={18}/>{completed} completed transcript{completed !== 1 ? 's' : ''}<span>View history <ArrowRight size={16}/></span></button>}
        </div><aside className="details-column">
          <section className="engine-card"><div className="card-kicker"><Cpu size={17}/>PROCESSING DEVICE</div>
            <label className="field-label" htmlFor="device">Processing device</label><select id="device" value={status?.settings.device || 'auto'} disabled={locked} onChange={e => void action('/settings', 'PUT', { device: e.target.value })}><option value="auto">Local GPU (if CUDA available)</option><option value="kaggle">Online GPU (Kaggle, free)</option><option value="cpu">CPU only</option></select>
            {online ? <div className="online-device">
              <h2>Online GPU</h2>
              <p className="engine-reason">Audio is processed on your private Kaggle account, deleted after completion, and saved to History. Kaggle includes 30 free GPU hours/week.</p>
              {!status?.kaggle.installed ? <><button className="small-button wide" disabled={busy || kaggleWorking} onClick={() => void action('/kaggle/download')}><ArrowDownToLine size={15}/>{kaggleWorking ? 'Downloading...' : 'Download Kaggle CLI'}</button><p className="download-size">Download: {bytes(status?.kaggle.download_bytes || 0)}</p></> : null}
              {status?.kaggle.status === 'downloading' && <progress max={1} value={status.kaggle.progress} aria-label="Kaggle CLI download progress"/>}
              {status?.kaggle.installed && status.kaggle.error && <button className="text-button" disabled={busy || kaggleWorking} onClick={() => void action('/kaggle/download')}>Repair Kaggle CLI</button>}
            </div> : <>
            <h2>{profile ? profile.device !== 'cpu' ? 'GPU' : 'CPU' : checking ? 'Checking device' : 'Awaiting model'}</h2>
            {status?.settings.device === 'cpu' && status.components.gpu.compatible && <p className="engine-reason">CPU processing may be significantly slower than GPU. For faster transcription, we recommend using a compatible GPU.</p>}
            {status?.settings.device === 'auto' && <>
              <p className="engine-reason" role="status">{status.components.gpu.reason}</p>
              {status.components.gpu.compatible && status.components.gpu.status === 'missing' && <>
                <button className="small-button wide" disabled={busy || status.components.tasks.some(t => t.id === 'gpu:auto' && ['queued','downloading','verifying','installing'].includes(t.status))} onClick={() => { if (confirm(`Download GPU Runtime (${bytes(status.components.gpu.download_bytes)})?`)) void action('/components/download','POST',{action:'download',component:'gpu:auto'}); }}><ArrowDownToLine size={15}/>Download GPU Runtime</button>
                <p className="download-size">Download: {bytes(status.components.gpu.download_bytes)}</p>
              </>}
              <button className="text-button" disabled={busy} onClick={() => void action('/components/recheck')}>Check GPU compatibility again</button>
            </>}
            </>}
            {(running || pausing) && <p className="engine-reason">Language and device are locked while the queue is running.</p>}
          </section>
          <section className="info-card model-card"><div className="card-kicker"><AudioLines size={17}/>TRANSCRIPTION MODEL</div>
            <label className="field-label" htmlFor="language">Language / Transcription model</label><select id="language" value={activeConfig?.language || status?.settings.language || 'fr'} disabled={locked} onChange={e => void action('/settings', 'PUT', { language: e.target.value })}>
              {status?.languages.map(model => <option key={model.language} value={model.language}>{model.label}</option>)}
            </select>
            <h3>{shownModel?.label || 'Selected'} transcription</h3><p className="model-id">{shownModel?.model || 'Loading model catalog...'}</p>
            {checking ? <p className="configuration-status" role="status"><LoaderCircle size={14} className="spin"/>Checking configuration...</p> : profile ? <div className="engine-stats"><div><span>Compute type</span><strong>{profile.compute_type}</strong></div><div><span>Concurrency</span><strong>{profile.concurrency}<small> chunks</small></strong></div></div> : null}
            {shownModel?.benchmark_status === 'pending' && <p className="benchmark-note">No confirmed optimal configuration yet.</p>}
            {!modelReady && <p role="status">{status?.model.message || 'Checking model files...'}</p>}
            {modelWorking && <progress max={1} value={status?.model.progress || 0} aria-label="Model download progress"/>}
            {!online && status?.model.error && <p className="job-error">{status.model.error}</p>}
            {!modelReady && <button className="small-button wide" disabled={modelWorking || busy || running || pausing || !status} onClick={() => void action('/model/download')}>{modelWorking ? <LoaderCircle className="spin" size={15}/> : <ArrowDownToLine size={15}/>}Download / repair model</button>}
            {!modelReady && status && !modelWorking && <p className="download-size">Download: {bytes(status.model.download_bytes)}</p>}
            {!online && status?.runtime.status === 'error' && <><details className="job-metadata"><summary>Configuration errors</summary>{status.runtime.attempts.map((a,i) => <p key={i}>{a.error}</p>)}</details><button className="small-button wide" disabled={locked} onClick={() => void action('/runtime/recheck')}><RefreshCw size={14}/>Check again</button></>}
          </section>
          {status && (status.components.tasks.length > 0 || status.components.setup_error) && <section className="info-card"><div className="card-kicker"><ArrowDownToLine size={17}/>OPTIONAL COMPONENTS</div>
            {status.components.setup_error && <p className="job-error">{status.components.setup_error}</p>}
            <button className="small-button" disabled={busy} onClick={() => void action('/components/control','POST',{action:status.components.paused ? 'resume':'pause'})}>{status.components.paused ? 'Resume downloads':'Pause downloads'}</button>
            {status.components.tasks.map(task => <div key={task.id} className="component-download"><h3>{task.label}</h3><p>{task.status === 'queued' && status.components.paused ? 'Paused':task.message}</p>
              {['downloading','verifying','installing'].includes(task.status) && <progress value={task.progress} max={1} aria-label={`${task.label} download progress`}/>}
              {task.error && <p className="job-error">{task.error}</p>}
              {['error','cancelled'].includes(task.status) ? <button className="small-button" disabled={busy} onClick={() => void action('/components/download','POST',{action:'download',component:task.id})}>Retry download</button> : task.status !== 'complete' && <button className="text-button" disabled={busy} onClick={() => void action('/components/control','POST',{action:'cancel',component:task.id})}>Cancel download</button>}
            </div>)}
          </section>}

        </aside></div>}
        {tab === 'history' && <section className="queue-card"><div className="section-heading"><h2>History <span className="count">{pastJobs.length}</span></h2><span className="muted">Saved until you delete it</span></div>{pastJobs.length ? pastJobs.map(renderJob) : <div className="empty-state"><History size={30}/><h3>No transcripts yet</h3><p>Completed transcripts will appear here.</p></div>}</section>}
      </main>
    </div>{exitOpen && <ExitDialog busy={busy} onClose={() => setExitOpen(false)} onChoose={choice => void chooseExit(choice)}/>}
    {preview && <Transcript job={preview} onClose={() => setPreview(null)}/>}
    {guideOpen && <UserGuide onClose={() => setGuideOpen(false)}/>}
  </div>;
}
