import { useEffect } from 'react';
import type { PendingCleanup } from '../types';

export function CleanupNotice({ job, onExpire }: { job: PendingCleanup; onExpire: (id: string) => void }) {
  useEffect(() => {
    const timer = setTimeout(() => onExpire(job.id), 5000);
    return () => clearTimeout(timer);
  }, [job.id, onExpire]);

  return (
    <div className="cleanup-banner" role="alert">
      <span>
        {job.name}: cleanup pending
        {job.owner ? ` in ${job.owner}'s ${job.provider === 'colab' ? 'Colab' : 'Kaggle'} account` : ''}.{' '}
        {Object.values(job.errors).join(' ') || 'Chep will retry cleanup automatically.'}
      </span>
    </div>
  );
}
