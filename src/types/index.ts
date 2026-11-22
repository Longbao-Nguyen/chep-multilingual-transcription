export type DevicePreference = 'kaggle' | 'colab' | 'auto' | 'cpu';

export type AppTab = 'workspace' | 'history';

export type ExitChoice = 'quit' | 'background';

export interface Profile {
  device: string;
  device_name: string;
  compute_type: string;
  concurrency: number;
  reason: string;
  language?: string;
  model_id?: string;
  chunk_seconds?: number;
}

export interface ModelConfig {
  label: string;
  model: string;
  ct2_model: string | null;
  backend: string;
  chunk_seconds: number;
  benchmark_status: string;
  validation_status: string;
}

export interface JobConfig {
  language: string;
  device_preference: string;
  model_config: ModelConfig;
  legacy?: boolean;
}

export interface Job {
  id: string;
  name: string;
  size: number;
  status: string;
  progress: number;
  duration: number | null;
  processed: number;
  error: string | null;
  profile: string | null;
  config: string | null;
  attempts: string | null;
  remote?: string | null;
}

export interface Remote {
  provider?: string;
  kernel?: string;
  dataset?: string;
  endpoint?: string;
  cleaned: boolean;
  cleanup_error?: string;
}

export interface PendingCleanup {
  id: string;
  name: string;
  owner?: string;
  provider?: string;
  errors: Record<string, string>;
}

export interface KaggleState {
  installed: boolean;
  status: string;
  username: string | null;
  message: string;
  progress: number;
  error: string | null;
  download_bytes: number;
}

export interface ColabState {
  installed: boolean;
  status: string;
  username: string | null;
  message: string;
  progress: number;
  error: string | null;
  download_bytes: number;
  auth_url: string | null;
}

export interface ComponentTask {
  id: string;
  label: string;
  status: string;
  progress: number;
  message: string;
  error: string | null;
}

export interface ComponentsState {
  gpu: {
    compatible: boolean;
    status: string;
    reason: string;
    download_bytes: number;
  };
  paused: boolean;
  setup_error?: string;
  tasks: ComponentTask[];
}

export interface QueueState {
  state: string;
  active_id: string | null;
  jobs: Job[];
  cleanup_pending: PendingCleanup[];
}

export interface KaggleQuota {
  used: string;
  remaining: string;
  total: string;
  refreshAt?: string;
}

export interface KaggleGpuState {
  status: string;
  username: string | null;
  message: string;
  quota: KaggleQuota | null;
}

export interface ColabQuota {
  balance: string;
  rate: string;
  active: number;
}

export interface ColabGpuState {
  status: string;
  username: string | null;
  message: string;
  quota: ColabQuota | null;
}

export interface ModelState {
  status: string;
  progress: number;
  message: string;
  error: string | null;
  download_bytes: number;
}

export interface RuntimeState {
  status: string;
  profile: Profile | null;
  message: string;
  attempts: { error: string }[];
}

export interface HardwareGpu {
  index: number;
  name: string;
  total_mib: number;
  free_mib: number;
}

export interface HardwareState {
  cpu_name: string;
  gpus: HardwareGpu[];
}

export interface SettingsState {
  device: string;
  language: string;
}

export interface Status {
  kaggle: KaggleState;
  kaggle_gpu: KaggleGpuState;
  colab: ColabState;
  colab_gpu: ColabGpuState;
  components: ComponentsState;
  queue: QueueState;
  model: ModelState;
  runtime: RuntimeState;
  languages: (ModelConfig & { language: string })[];
  hardware: HardwareState;
  settings: SettingsState;
  data_dir: string;
  free_bytes: number;
  version: string;
}

export interface UploadState {
  name: string;
  progress: number;
  index: number;
  total: number;
}
