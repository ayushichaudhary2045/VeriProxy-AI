let silenceStartTime: number | null = null;
let pitchHistory: number[] = [];
const SILENCE_THRESHOLD = 0.015; 
const LATENT_SILENCE_LIMIT_MS = 2500; 
const MONOTONIC_VARIANCE_THRESHOLD = 5.0; 

export async function initVoiceProctoring(
  stream: MediaStream, 
  onRiskBreach: (reason: string, metadata: any) => void
) {
  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  
  // Point directly to public/audio-processor.js
  await audioCtx.audioWorklet.addModule('/audio-processor.js');
  const workletNode = new AudioWorkletNode(audioCtx, 'voice-metrics-processor');
  
  source.connect(workletNode);

  workletNode.port.onmessage = (event) => {
    const { rms, samples } = event.data;
    
    // Latent Silence Check
    if (rms < SILENCE_THRESHOLD) {
      if (!silenceStartTime) silenceStartTime = Date.now();
      const durationMs = Date.now() - silenceStartTime;
      
      if (durationMs > LATENT_SILENCE_LIMIT_MS) {
        onRiskBreach("LATENT_SILENCE_DETECTED", { durationMs });
        silenceStartTime = Date.now();
      }
    } else {
      silenceStartTime = null; 
      
      // Pitch Extraction
      const pitch = autoCorrelatePitch(samples, audioCtx.sampleRate);
      if (pitch > 50 && pitch < 500) { 
        pitchHistory.push(pitch);
        if (pitchHistory.length > 50) pitchHistory.shift(); 
        
        const stdDev = calculateStdDev(pitchHistory);
        if (pitchHistory.length >= 30 && stdDev < MONOTONIC_VARIANCE_THRESHOLD) {
          onRiskBreach("UNNATURAL_MONOTONIC_VOICE", { variance: stdDev });
          pitchHistory = []; 
        }
      }
    }
  };
}

function autoCorrelatePitch(buffer: number[], sampleRate: number): number {
  let size = buffer.length;
  let r = new Array(size).fill(0);
  for (let lag = 0; lag < size; lag++) {
    for (let i = 0; i < size - lag; i++) {
      r[lag] += buffer[i] * buffer[i + lag];
    }
  }
  let maxLag = -1, maxVal = -1;
  for (let lag = 20; lag < size / 2; lag++) {
    if (r[lag] > maxVal) {
      maxVal = r[lag];
      maxLag = lag;
    }
  }
  return maxLag !== -1 ? sampleRate / maxLag : -1;
}

function calculateStdDev(arr: number[]): number {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / arr.length);
}