import type { RefObject } from 'react';
import { Plus, Upload } from 'lucide-react';
import type { Status, UploadState } from '../types';

interface UploadDropzoneProps {
  inputRef: RefObject<HTMLInputElement | null>;
  dragging: boolean;
  upload: UploadState | null;
  selectionPending: boolean;
  status: Status | null;
  online: boolean;
  onDraggingChange: (dragging: boolean) => void;
  onFilesSelected: (files: File[]) => void;
}

export function UploadDropzone({
  inputRef,
  dragging,
  upload,
  selectionPending,
  status,
  online,
  onDraggingChange,
  onFilesSelected,
}: UploadDropzoneProps) {
  return (
    <section
      className={`upload-card ${dragging ? 'dragging' : ''}`}
      onDragOver={e => {
        e.preventDefault();
        onDraggingChange(true);
      }}
      onDragLeave={() => onDraggingChange(false)}
      onDrop={e => {
        e.preventDefault();
        onDraggingChange(false);
        onFilesSelected(Array.from(e.dataTransfer.files));
      }}
    >
      <span className="upload-icon">
        <Upload size={26} />
      </span>
      <h2>Add your audio</h2>
      <p>Drag files here, or choose them from your computer.</p>
      <button
        className="primary-button"
        disabled={Boolean(upload) || selectionPending || !status}
        onClick={() => inputRef.current?.click()}
      >
        <Plus size={17} />
        Choose files
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        hidden
        accept="audio/*,video/*,.m4a,.mkv,.wma,.opus,.flac"
        onChange={e => onFilesSelected(Array.from(e.target.files || []))}
      />
      <small>MP3, WAV, M4A, MP4 and other common formats · up to {online ? '2' : '20'} GB per file</small>
      {upload && (
        <div className="upload-progress" role="status">
          <span>
            {upload.index}/{upload.total} · {upload.name}
          </span>
          <progress max={1} value={upload.progress} />
          <small>Copying to local storage · {Math.round(upload.progress * 100)}%</small>
        </div>
      )}
    </section>
  );
}
