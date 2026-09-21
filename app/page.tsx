'use client';

import { useState } from 'react';
import { initVoiceProctoring } from './utils/voice-detector';

export default function Home() {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [alerts, setAlerts] = useState<string[]>([]);

  const TRIGGER_COOLDOWN_MS = 5000; // Wait 5 seconds between alerts to prevent spam

  const startProctoring = async () => {
    try {
      let lastTriggerTime = 0;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsMonitoring(true);

      await initVoiceProctoring(stream, (reason, metadata) => {
        const now = Date.now();
        
        // Only log if 5 seconds have passed since the last alert
        if (now - lastTriggerTime > TRIGGER_COOLDOWN_MS) {
          lastTriggerTime = now;
          const time = new Date().toLocaleTimeString();
          const alertMsg = `[${time}] Risk Triggered: ${reason}`;
          
          setAlerts((prev) => [alertMsg, ...prev]);
          sendBurstToGemini(reason, metadata);
        }
      });
    } catch (err) {
      console.error("Microphone access denied or error occurred:", err);
      alert("Please allow microphone access to start voice proctoring!");
    }
  }; 
  
  const sendBurstToGemini = async (reason: string, metadata: any) => {
    console.log("Sending burst log to Gemini API for reason:", reason, metadata);
    // Future step: Here we send the 5-second snapshot to our Next.js API endpoint
  };

  return (
    <main style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>VeriProxy AI - Voice Proctoring</h1>
      <p>Status: <strong>{isMonitoring ? '🟢 Active' : '🔴 Inactive'}</strong></p>

      {!isMonitoring && (
        <button 
          onClick={startProctoring}
          style={{
            padding: '10px 20px',
            backgroundColor: '#0070f3',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '16px'
          }}
        >
          Start Voice Proctoring
        </button>
      )}

      <div style={{ marginTop: '2rem' }}>
        <h2>Live Risk Alerts</h2>
        {alerts.length === 0 ? (
          <p style={{ color: '#888' }}>No threats detected yet. Speak normally into your microphone!</p>
        ) : (
          <ul style={{ color: 'red' }}>
            {alerts.map((alert, index) => (
              <li key={index} style={{ marginBottom: '5px' }}>{alert}</li>
            ))}
          </ul>
        )}
      </div>
    </main>
  );
}