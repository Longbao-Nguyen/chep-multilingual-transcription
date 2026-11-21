import { LoaderCircle } from 'lucide-react';
import type { ExitChoice } from '../types';

export function GoodbyeScreen({
  departure,
  serverStopped,
  onReturn,
}: {
  departure: ExitChoice;
  serverStopped: boolean;
  onReturn: () => void;
}) {
  return (
    <main className="goodbye">
      <h1>
        {departure === 'background'
          ? 'Chep is running in background.'
          : serverStopped
            ? 'Chep has stopped.'
            : 'Quitting Chep...'}
      </h1>
      <p>
        {departure === 'background'
          ? 'Transcription and downloads can continue. You can close this tab.'
          : serverStopped
            ? 'Application files and local resources have been released. You can close this tab.'
            : 'Stopping local work and saving your queue. Please wait before updating or removing the app.'}
      </p>
      {departure === 'background' && (
        <button className="primary-button" onClick={onReturn}>
          Return to Chep
        </button>
      )}
      {departure === 'quit' && !serverStopped && <LoaderCircle className="spin" />}
    </main>
  );
}
