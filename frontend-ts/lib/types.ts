export interface SearchResult {
  timestamp_ms: number;
  score: number;
  labels: string[];
  box?: [number, number, number, number]; // [x, y, w, h] in a 224x224 frame
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface UploadResponse {
  video_id: string;
  status: string;
}

export interface StatusResponse {
  video_id: string;
  frames: number;
  done: boolean;
  error?: string;
}
