import type { ExitChoice, Job, Status } from '../types';

function initialToken(): string {
  const token = new URLSearchParams(location.hash.slice(1)).get('token');
  if (token) {
    sessionStorage.setItem('ft-token', token);
    history.replaceState(null, '', location.pathname);
  }
  return token || sessionStorage.getItem('ft-token') || '';
}

export const authToken = initialToken();

export async function apiRequest(path: string, options: RequestInit = {}): Promise<Response> {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { Authorization: `Bearer ${authToken}`, ...options.headers },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : 'The action could not be completed.');
  }
  return response;
}

export async function fetchStatus(signal?: AbortSignal): Promise<Status> {
  const response = await apiRequest('/status', { signal });
  return (await response.json()) as Status;
}

export async function fetchTranscript(
  jobId: string,
  format: 'txt' | 'srt' = 'txt',
  signal?: AbortSignal,
): Promise<Response> {
  const query = format === 'txt' ? '' : `?format=${format}`;
  return apiRequest(`/jobs/${jobId}/transcript${query}`, { signal });
}

export async function postLifecycleChoice(choice: ExitChoice): Promise<Response> {
  return apiRequest('/lifecycle/choose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ choice }),
  });
}

export function uploadJobFile(file: File, onProgress: (progress: number) => void): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/jobs?name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader('Authorization', `Bearer ${authToken}`);
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) {
        onProgress(event.loaded / event.total);
      }
    };
    xhr.onerror = () => reject(new Error('Connection lost while adding the file.'));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        let message = 'Could not add the file.';
        try {
          message = JSON.parse(xhr.responseText).detail || message;
        } catch {
          // Keep default error message when response body is not JSON.
        }
        reject(new Error(message));
      }
    };
    xhr.send(file);
  });
}

export async function downloadTranscriptFile(job: Job, format: 'txt' | 'srt'): Promise<void> {
  const response = await apiRequest(`/jobs/${job.id}/transcript?format=${format}`);
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${job.name.replace(/\.[^.]+$/, '')}.${format}`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export function getUiSessionSocketUrl(): string {
  return `ws://${location.host}/api/ui-session`;
}
