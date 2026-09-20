"use client";

import { useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

interface ViolationLog {
  id: number;
  time: string;
  message: string;
  points: number;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const animationRef = useRef<number | null>(null);

  const sideGazeFramesRef = useRef<number>(0);
  const verticalGazeFramesRef = useRef<number>(0);

  const lastSideDropRef = useRef<number>(0);
  const lastVerticalDropRef = useRef<number>(0);

  const handTouchStartRef = useRef<number | null>(null);
  const lastHandDropRef = useRef<number>(0);

  const noFaceSinceRef = useRef<number | null>(null);
  const lastTimestampRef = useRef<number>(0);

  const [cameraOn, setCameraOn] = useState<boolean>(false);
  const [score, setScore] = useState<number>(100);

  const [analysisStatus, setAnalysisStatus] = useState<string>(
    "Waiting for behavioral analysis..."
  );
  const [logs, setLogs] = useState<ViolationLog[]>([]);

  const statusRef = useRef<string>("Waiting for behavioral analysis...");

  useEffect(() => {
    if (typeof window !== "undefined") {
      const originalError = console.error;
      console.error = (...args: unknown[]) => {
        if (
          typeof args[0] === "string" &&
          args[0].includes("INFO: Created TensorFlow Lite")
        ) {
          return;
        }
        originalError.apply(console, args);
      };

      return () => {
        console.error = originalError;
      };
    }
  }, []);

  const getGaugeColor = (val: number) => {
    if (val >= 80)
      return {
        border: "border-green-500",
        text: "text-green-500",
        bg: "bg-green-500/10",
      };
    if (val >= 50)
      return {
        border: "border-yellow-500",
        text: "text-yellow-500",
        bg: "bg-yellow-500/10",
      };
    return {
      border: "border-red-500",
      text: "text-red-500",
      bg: "bg-red-500/10",
    };
  };

  const gaugeStyle = getGaugeColor(score);

  const applyRiskAdjustment = (change: number, message: string) => {
    setScore((currentScore) => Math.max(0, Math.min(100, currentScore - change)));

    statusRef.current = message;
    setAnalysisStatus(message);

    const timeStr = new Date().toLocaleTimeString();
    setLogs((prev) => [
      { id: Date.now(), time: timeStr, message, points: change },
      ...prev,
    ]);
  };

  const stopCameraStream = () => {
    const video = videoRef.current;
    if (!video) return;

    const stream = video.srcObject as MediaStream | null;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    video.pause();
    video.srcObject = null;
  };

  const startCamera = async () => {
    stopCameraStream();

    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    try {
      setAnalysisStatus("Starting camera...");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });

      const video = videoRef.current;
      if (!video) return;

      video.srcObject = stream;
      await video.play();

      setCameraOn(true);
      setAnalysisStatus("Loading Vision Models...");

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
      });

      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });

      setAnalysisStatus("Behavioral analysis active");
      detectFaceAndHands();
    } catch (error) {
      console.error("Initialization error:", error);
      setAnalysisStatus("Error loading models or camera.");
    }
  };

  const detectFaceAndHands = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const faceLandmarker = faceLandmarkerRef.current;
    const handLandmarker = handLandmarkerRef.current;

    if (!video || !faceLandmarker || !handLandmarker || !canvas) {
      animationRef.current = requestAnimationFrame(detectFaceAndHands);
      return;
    }

    if (video.readyState < 2) {
      animationRef.current = requestAnimationFrame(detectFaceAndHands);
      return;
    }

    const ctx = canvas.getContext("2d");
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    try {
      const timestamp = Math.max(performance.now(), lastTimestampRef.current + 1);
      lastTimestampRef.current = timestamp;

      const faceResults = faceLandmarker.detectForVideo(video, timestamp);
      const handResults = handLandmarker.detectForVideo(video, timestamp);

      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      let faceBox = { minX: 1, maxX: 0, minY: 1, maxY: 0 };

      if (faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
        const landmarks = faceResults.faceLandmarks[0];
        noFaceSinceRef.current = null;

        if (ctx) {
          const drawingUtils = new DrawingUtils(ctx);
          
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { color: "#C0C0C040", lineWidth: 1 }
          );

          // Green Iris Ring & Centers
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
            { color: "#00FF66", lineWidth: 2 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
            { color: "#00FF66", lineWidth: 2 }
          );

          [468, 473].forEach((index) => {
            const point = landmarks[index];
            if (point) {
              ctx.beginPath();
              ctx.arc(
                point.x * canvas.width,
                point.y * canvas.height,
                4,
                0,
                2 * Math.PI
              );
              ctx.fillStyle = "#00FF66";
              ctx.fill();
            }
          });
        }

        // Bounding Box Calculation
        landmarks.forEach((pt) => {
          if (pt.x < faceBox.minX) faceBox.minX = pt.x;
          if (pt.x > faceBox.maxX) faceBox.maxX = pt.x;
          if (pt.y < faceBox.minY) faceBox.minY = pt.y;
          if (pt.y > faceBox.maxY) faceBox.maxY = pt.y;
        });

        const nose = landmarks[1];
        const leftCheek = landmarks[234];
        const rightCheek = landmarks[454];
        const forehead = landmarks[10];
        const chin = landmarks[152];

        const now = Date.now();

        // --- 1. SIDE MOVEMENT DETECTION ---
        const faceWidth = Math.abs(rightCheek.x - leftCheek.x);
        const noseRatioH =
          faceWidth > 0.001 ? (nose.x - leftCheek.x) / faceWidth : 0.5;

        const isTurningSide = noseRatioH < 0.32 || noseRatioH > 0.68;

        if (isTurningSide) {
          sideGazeFramesRef.current += 1;
        } else {
          sideGazeFramesRef.current = Math.max(0, sideGazeFramesRef.current - 1);
        }

        if (sideGazeFramesRef.current >= 3 && now - lastSideDropRef.current > 2000) {
          lastSideDropRef.current = now;
          sideGazeFramesRef.current = 0;
          applyRiskAdjustment(5, "Gaze Violation: Looking to the side");
        }

        // --- 2. UPWARD & DOWNWARD GAZE MOVEMENT DETECTION ---
        const faceHeight = Math.abs(chin.y - forehead.y);
        const noseRatioV =
          faceHeight > 0.001 ? (nose.y - forehead.y) / faceHeight : 0.5;

        // < 0.32 means Head/Gaze Up | > 0.62 means Head/Gaze Down
        const isLookingUpOrDown = noseRatioV < 0.32 || noseRatioV > 0.62;

        if (isLookingUpOrDown) {
          verticalGazeFramesRef.current += 1;
        } else {
          verticalGazeFramesRef.current = Math.max(0, verticalGazeFramesRef.current - 1);
        }

        if (
          verticalGazeFramesRef.current >= 3 &&
          now - lastVerticalDropRef.current > 2000
        ) {
          lastVerticalDropRef.current = now;
          verticalGazeFramesRef.current = 0;
          const direction = noseRatioV < 0.32 ? "upward" : "downward";
          applyRiskAdjustment(5, `Gaze Violation: Looking ${direction}`);
        }

      } else {
        const now = Date.now();
        if (noFaceSinceRef.current === null) noFaceSinceRef.current = now;
        if (now - noFaceSinceRef.current > 2000) {
          applyRiskAdjustment(5, "Face lost — attention risk");
          noFaceSinceRef.current = now;
        }
      }

      // --- 3. HAND OVER FACE DETECTION ---
      let isHandTouchingFace = false;

      if (
        handResults.landmarks &&
        handResults.landmarks.length > 0 &&
        ctx
      ) {
        const drawingUtils = new DrawingUtils(ctx);

        handResults.landmarks.forEach((handLandmarks) => {
          drawingUtils.drawConnectors(
            handLandmarks,
            HandLandmarker.HAND_CONNECTIONS,
            { color: "#FF0055", lineWidth: 2 }
          );

          handLandmarks.forEach((pt) => {
            if (
              pt.x >= faceBox.minX - 0.05 &&
              pt.x <= faceBox.maxX + 0.05 &&
              pt.y >= faceBox.minY - 0.05 &&
              pt.y <= faceBox.maxY + 0.05
            ) {
              isHandTouchingFace = true;
            }
          });
        });
      }

      const now = Date.now();

      if (isHandTouchingFace) {
        if (handTouchStartRef.current === null) {
          handTouchStartRef.current = now;
        }

        if (
          now - handTouchStartRef.current > 400 &&
          now - lastHandDropRef.current > 2500
        ) {
          lastHandDropRef.current = now;
          applyRiskAdjustment(5, "Hand covering face/mouth detected");
        }
      } else {
        handTouchStartRef.current = null;
      }
    } catch (error) {
      console.error("Detection Loop Error:", error);
    }

    animationRef.current = requestAnimationFrame(detectFaceAndHands);
  };

  useEffect(() => {
    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
      stopCameraStream();
    };
  }, []);

  return (
    <main className="min-h-screen bg-[#050816] text-white">
      <header className="border-b border-white/10 px-8 py-5">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">VeriProxy-AI</h1>
            <p className="text-sm text-gray-400">
              AI-Powered Behavioral Integrity Monitor
            </p>
          </div>
          <div
            className={`rounded-full border px-4 py-2 text-sm ${gaugeStyle.bg} ${gaugeStyle.text} border-current`}
          >
            ● System Active
          </div>
        </div>
      </header>

      <section className="mx-auto grid max-w-6xl gap-8 px-8 py-10 md:grid-cols-2">
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Candidate Camera</h2>
            <span className="text-sm text-gray-400">
              {cameraOn ? "Camera Active" : "Camera Off"}
            </span>
          </div>

          <div className="relative aspect-video overflow-hidden rounded-xl bg-black">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="h-full w-full object-cover -scale-x-100"
            />
            <canvas
              ref={canvasRef}
              className="pointer-events-none absolute inset-0 h-full w-full object-cover -scale-x-100"
            />
          </div>

          <button
            onClick={startCamera}
            disabled={cameraOn}
            className="mt-5 w-full rounded-xl bg-blue-600 px-5 py-3 font-semibold transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {cameraOn ? "Camera Active" : "Start Camera"}
          </button>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 flex flex-col justify-between">
          <div>
            <h2 className="text-lg font-semibold">Integrity Score</h2>

            <div className="mt-6 flex flex-col items-center">
              <div
                className={`flex h-44 w-44 items-center justify-center rounded-full border-8 transition-colors duration-500 ${gaugeStyle.border} ${gaugeStyle.bg}`}
              >
                <div className="text-center">
                  <div
                    className={`text-5xl font-bold transition-colors duration-500 ${gaugeStyle.text}`}
                  >
                    {score}
                  </div>
                  <div className="mt-1 text-xs text-gray-400">Risk Score</div>
                </div>
              </div>

              <p className="mt-4 text-center text-sm text-gray-300">
                {analysisStatus}
              </p>
            </div>
          </div>

          <div className="mt-6 rounded-xl border border-white/10 bg-black/40 p-4">
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">
              Security Feed
            </h3>
            <div className="max-h-32 overflow-y-auto space-y-2 pr-1">
              {logs.length === 0 ? (
                <p className="text-xs text-gray-500">No flags recorded yet.</p>
              ) : (
                logs.map((log) => (
                  <div
                    key={log.id}
                    className="flex items-center justify-between text-xs rounded border border-red-500/20 bg-red-500/10 px-2.5 py-1.5 text-red-300"
                  >
                    <span>
                      [{log.time}] {log.message}
                    </span>
                    <span className="font-bold text-red-400">
                      -{log.points} pts
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}