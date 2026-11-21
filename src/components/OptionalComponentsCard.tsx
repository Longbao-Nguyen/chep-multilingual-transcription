import { ArrowDownToLine } from 'lucide-react';
import type { ComponentsState } from '../types';

interface OptionalComponentsCardProps {
  components: ComponentsState;
  busy: boolean;
  onAction: (path: string, method?: string, body?: object) => void;
}

export function OptionalComponentsCard({ components, busy, onAction }: OptionalComponentsCardProps) {
  return (
    <section className="info-card">
      <div className="card-kicker">
        <ArrowDownToLine size={17} />
        OPTIONAL COMPONENTS
      </div>
      {components.setup_error && <p className="job-error">{components.setup_error}</p>}
      <button
        className="small-button"
        disabled={busy}
        onClick={() =>
          onAction('/components/control', 'POST', { action: components.paused ? 'resume' : 'pause' })
        }
      >
        {components.paused ? 'Resume downloads' : 'Pause downloads'}
      </button>
      {components.tasks.map(task => (
        <div key={task.id} className="component-download">
          <h3>{task.label}</h3>
          <p>{task.status === 'queued' && components.paused ? 'Paused' : task.message}</p>
          {['downloading', 'verifying', 'installing'].includes(task.status) && (
            <progress value={task.progress} max={1} aria-label={`${task.label} download progress`} />
          )}
          {task.error && <p className="job-error">{task.error}</p>}
          {['error', 'cancelled'].includes(task.status) ? (
            <button
              className="small-button"
              disabled={busy}
              onClick={() =>
                onAction('/components/download', 'POST', { action: 'download', component: task.id })
              }
            >
              Retry download
            </button>
          ) : (
            task.status !== 'complete' && (
              <button
                className="text-button"
                disabled={busy}
                onClick={() =>
                  onAction('/components/control', 'POST', { action: 'cancel', component: task.id })
                }
              >
                Cancel download
              </button>
            )
          )}
        </div>
      ))}
    </section>
  );
}
