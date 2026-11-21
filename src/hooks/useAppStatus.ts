import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchStatus } from '../services/api';
import type { ExitChoice, Status } from '../types';

export function useAppStatus(departure: ExitChoice | null) {
  const [status, setStatus] = useState<Status | null>(null);
  const [connectionError, setConnectionError] = useState('');
  const mounted = useRef(true);
  const epoch = useRef(0);

  const invalidateEpoch = useCallback(() => {
    epoch.current++;
  }, []);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    const generation = epoch.current;
    const next = await fetchStatus(signal);
    if (mounted.current && generation === epoch.current) {
      setStatus(next);
      setConnectionError('');
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    if (departure) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        await refresh(abort.signal);
      } catch (e) {
        if (!abort.signal.aborted) {
          setConnectionError(e instanceof Error ? e.message : 'Connection lost.');
        }
      }
      if (!abort.signal.aborted) {
        timer = setTimeout(poll, 1500);
      }
    };
    void poll();
    return () => {
      mounted.current = false;
      abort.abort();
      clearTimeout(timer);
    };
  }, [refresh, departure]);

  return { status, connectionError, refresh, invalidateEpoch };
}
