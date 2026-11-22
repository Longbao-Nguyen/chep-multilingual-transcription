import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

export function UserGuideDrawer({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.current?.showModal();
    return () => {
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, []);

  return (
    <dialog
      ref={dialog}
      id="user-guide"
      className="guide-drawer"
      aria-labelledby="guide-title"
      onCancel={e => {
        e.preventDefault();
        onClose();
      }}
    >
      <header>
        <h2 id="guide-title">Chep — User Guide</h2>
        <button className="icon-button" aria-label="Close user guide" onClick={onClose} autoFocus>
          <X size={22} />
        </button>
      </header>
      <div className="guide-content">
        <section>
          <h3>1. Common Use</h3>
          <h4>Step 1: Select a Processing Device</h4>
          <p>Each processing option has its own advantages:</p>
          <ul>
            <li>
              <strong>Online GPU (Recommended):</strong> Kaggle typically provides up to 30 hours of free GPU usage per week. The models were benchmarked and optimized on Kaggle GPUs during development, making this the recommended option for most users.
            </li>
            <li>
              <strong>Local GPU:</strong> Provides greater privacy by keeping all input audio files and output transcripts on your device. This option is available only for compatible NVIDIA GPUs with CUDA support.
            </li>
            <li>
              <strong>CPU Only:</strong> Offers the same level of privacy as Local GPU but processes audio significantly more slowly (approximately 25× slower, depending on the hardware and model). Best suited for short audio files.
            </li>
          </ul>
          <h4>Step 2: Select a Language</h4>
          <p>
            Choose the language of the audio you want to transcribe. This version of Chep supports French, English, Spanish, and Vietnamese.
          </p>
          <h4>Step 3: Add Audio Files to the Queue</h4>
          <p>Select or drag and drop the audio files you want to transcribe to create a processing queue.</p>
          <ul>
            <li>
              Click <strong>Start</strong> to process the files sequentially.
            </li>
            <li>You can remove individual files from the queue without affecting other files.</li>
            <li>
              You also can cancel an active transcription task if needed. This will pause the queue until you click{' '}
              <strong>Start</strong> again.
            </li>
          </ul>
          <h4>Step 4: Access Your Transcripts</h4>
          <p>
            Once processing is complete, your transcripts will appear in the <strong>History</strong> tab, where you can view and download them.
          </p>
        </section>
        <section>
          <h3>2. Setup &amp; Installation</h3>
          <h4>CPU Only</h4>
          <p>
            CPU processing is available by default and does not require a dedicated GPU. However, you must download the appropriate language model before running local transcription.
          </p>
          <h4>Local GPU</h4>
          <p>Local GPU processing requires a compatible NVIDIA GPU. You need to:</p>
          <ol>
            <li>Download the required CUDA runtime components.</li>
            <li>Download the language models you want to use.</li>
          </ol>
          <p>Chep will be able to run transcription directly on your device when the required components are installed.</p>
          <h4>Online GPU (Kaggle — Recommended)</h4>
          <p>
            Online GPU is a convenient option that allows you to use Kaggle's free GPU resources without requiring powerful local hardware.
          </p>
          <ol>
            <li>
              Click <strong>Download Kaggle CLI</strong>.
            </li>
            <li>Create a Kaggle account or sign in to your existing account.</li>
            <li>
              Click <strong>Authenticate Kaggle CLI</strong> and approve the authorization request.
            </li>
            <li>
              Click <strong>Check GPU condition</strong> and verify your phone number to access GPU.
            </li>
          </ol>
          <p>
            Once finished, you can start transcribing without downloading language models or CUDA components to your computer.
          </p>
          <p>
            <strong>Note:</strong> Online GPU processing requires an internet connection. Your audio files will be uploaded to your private Kaggle account for processing and deleted after.
          </p>
        </section>
        <section>
          <h3>3. Important Notes</h3>
          <p>
            <strong>Always follow the steps in the correct order:</strong>
          </p>
          <ol>
            <li>Select your processing device.</li>
            <li>Select your transcription language and model.</li>
            <li>Add your audio files to the queue.</li>
            <li>
              Click <strong>Start</strong> to begin transcription.
            </li>
          </ol>
          <p>
            <strong>Processing device and model selections are locked for each file once it is added to the queue.</strong> Changing these settings afterward will only affect newly added files. To use a different processing device or model for an existing file, remove that file from the queue and add it again with the desired settings.
          </p>
        </section>
        <p>
          You can follow progress on Kaggle: click <strong>Create/Sign in an account</strong>, then click{' '}
          <strong>View Active Events</strong> in the bottom-left corner.
        </p>
      </div>
    </dialog>
  );
}
