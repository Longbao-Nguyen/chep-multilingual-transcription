export const ACTIVE_JOB_STATUSES = new Set([
  'LOADING',
  'PREPROCESSING',
  'TRANSCRIBING',
  'FINALIZING',
  'UPLOADING',
  'WAITING_GPU',
  'REMOTE_RUNNING',
  'DOWNLOADING',
]);

export const JOB_STATUS_LABELS: Record<string, string> = {
  UPLOADING: 'Uploading to Kaggle',
  WAITING_GPU: 'Waiting for Kaggle GPU',
  REMOTE_RUNNING: 'Processing on Kaggle',
  DOWNLOADING: 'Retrieving transcript',
  QUEUED: 'Queued',
  LOADING: 'Loading model',
  PREPROCESSING: 'Preparing audio',
  TRANSCRIBING: 'Transcribing',
  FINALIZING: 'Saving',
  COMPLETED: 'Completed',
  FAILED: 'Failed',
  INTERRUPTED: 'Interrupted',
  CANCELLED: 'Cancelled',
};

const COLAB_STATUS_LABELS: Record<string, string> = {
  UPLOADING: 'Uploading to Colab',
  WAITING_GPU: 'Waiting for Colab GPU',
  REMOTE_RUNNING: 'Processing on Colab',
};

export function jobStatusLabel(status: string, devicePreference?: string): string {
  if (devicePreference === 'colab' && status in COLAB_STATUS_LABELS) {
    return COLAB_STATUS_LABELS[status];
  }
  return JOB_STATUS_LABELS[status] || status;
}

export function formatBytes(n: number): string {
  return n >= 1024 ** 3
    ? `${(n / 1024 ** 3).toFixed(1)} GB`
    : `${(n / 1024 ** 2).toFixed(1)} MB`;
}

export function formatDuration(n: number): string {
  return `${Math.floor(n / 60)}:${Math.floor(n % 60).toString().padStart(2, '0')}`;
}

export function parseJson<T>(value: string | null | undefined): T | undefined {
  try {
    return value ? (JSON.parse(value) as T) : undefined;
  } catch {
    return undefined;
  }
}

export function formatErrorText(value: string): string {
  return /[\u0102-\u01b0\u1ea0-\u1ef9]/i.test(value)
    ? 'This file failed in an earlier app version. Retry it or check its original log for details.'
    : value;
}
