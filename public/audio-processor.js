class VoiceMetricsProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channelData = input[0];
      let sum = 0;
      for (let i = 0; i < channelData.length; i++) {
        sum += channelData[i] * channelData[i];
      }
      const rms = Math.sqrt(sum / channelData.length);
      
      this.port.postMessage({
        rms: rms,
        samples: Array.from(channelData)
      });
    }
    return true;
  }
}

registerProcessor('voice-metrics-processor', VoiceMetricsProcessor);