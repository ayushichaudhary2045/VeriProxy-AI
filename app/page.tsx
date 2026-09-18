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

function horizontalRatio(
  cornerA: { x: number },
  cornerB: { x: number },
  iris: { x: number }
) {
  const min = Math.min(cornerA.x, cornerB.x);
  const max = Math.max(cornerA.x, cornerB.x);
  if (max - min < 0.001) return 0.5;
  return (iris.x - min) / (max - min);
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
  const lastDetectionRef = useRef<number>(0);

  const [cameraOn, setCameraOn] = useState(false);
  const [score, setScore] = useState(100);

  const [analysisStatus, setAnalysisStatus] = useState(
    "Waiting for behavioral analysis..."
  );
  const [logs, setLogs] = useState<ViolationLog[]>([]);

  const statusRef = useRef("Waiting for behavioral analysis...");

  useEffect(() => {
    if (typeof window !== "undefined") {
      const originalError = console.error;
      console.error = (...args: any[]) => {
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

  useEffect(() => {
    const checkConnectedDevices = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();

        const foreignDevice = devices.find((device) => {
          const label = device.label.toLowerCase();
          return (
            label.includes("bluetooth") ||
            label.includes("airpods") ||
            label.includes("headset") ||
            label.includes("wireless") ||
            label.includes("buds") ||
            label.includes("hands-free")
          );
        });

        if (foreignDevice) {
          applyRiskAdjustment(
            10,
            `External Audio Connected: ${
              foreignDevice.label || "Wireless/Bluetooth Device"
            }`
          );
        }
      } catch (err) {
        console.error("Device detection error:", err);
      }
    };

    checkConnectedDevices();
    navigator.mediaDevices.addEventListener("devicechange", checkConnectedDevices);

    return () => {
      navigator.mediaDevices.removeEventListener(
        "devicechange",
        checkConnectedDevices
      );
    };
  }, []);

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

    if (faceLandmarkerRef.current) {
      faceLandmarkerRef.current.close();
      faceLandmarkerRef.current = null;
    }

    if (handLandmarkerRef.current) {
      handLandmarkerRef.current.close();
      handLandmarkerRef.current = null;
    }

    try {
      setAnalysisStatus("Starting camera...");

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is not supported in this browser.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: true,
      });

      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("Video element was not found.");
      }

      video.srcObject = stream;

      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          reject(new Error("Camera video did not become ready."));
        }, 10000);

        const checkVideo = () => {
          if (
            video.readyState >= 2 &&
            video.videoWidth > 0 &&
            video.videoHeight > 0
          ) {
            window.clearTimeout(timeout);
            resolve();
          }
        };

        video.onloadedmetadata = checkVideo;
        video.onloadeddata = checkVideo;
        checkVideo();
      });

      await video.play();

      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });

      setCameraOn(true);
      setAnalysisStatus("Initializing Face & Hand Detection...");

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      let faceLandmarker: FaceLandmarker;
      try {
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch (faceErr) {
        console.error("FaceLandmarker failed to load:", faceErr);
        throw new Error(
          "Face detection model failed to load — check your internet connection."
        );
      }
      faceLandmarkerRef.current = faceLandmarker;

      let handLandmarker: HandLandmarker;
      try {
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch (handErr) {
        console.error("HandLandmarker failed to load:", handErr);
        throw new Error(
          "Hand detection model failed to load — check your internet connection."
        );
      }
      handLandmarkerRef.current = handLandmarker;

      sideGazeFramesRef.current = 0;
      verticalGazeFramesRef.current = 0;
      lastSideDropRef.current = 0;
      lastVerticalDropRef.current = 0;

      handTouchStartRef.current = null;
      lastHandDropRef.current = 0;

      noFaceSinceRef.current = null;
      lastTimestampRef.current = 0;
      lastDetectionRef.current = 0;

      statusRef.current = "Behavioral analysis active";
      setAnalysisStatus("Behavioral analysis active");
      console.log("✅ Both models loaded — detection loop starting now.");

      startDetection();
    } catch (error) {
      console.error("Camera / MediaPipe error:", error);
      setCameraOn(false);

      const isCameraBusy =
        error instanceof DOMException && error.name === "NotReadableError";
      const isPermissionDenied =
        error instanceof DOMException && error.name === "NotAllowedError";

      let message = "Unable to start analysis.";
      if (isCameraBusy) {
        message =
          "Camera is already in use. Close other apps using the camera and try again.";
      } else if (isPermissionDenied) {
        message =
          "Camera/Audio permission was denied. Allow permissions and try again.";
      } else if (error instanceof Error) {
        message = error.message;
      }

      setAnalysisStatus(message);
      alert(message);
    }
  };

  const startDetection = () => {
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
    }
    animationRef.current = requestAnimationFrame(detectFaceAndHands);
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

    if (
      video.readyState < 2 ||
      video.videoWidth <= 0 ||
      video.videoHeight <= 0 ||
      video.paused ||
      video.ended
    ) {
      animationRef.current = requestAnimationFrame(detectFaceAndHands);
      return;
    }

    const ctx = canvas.getContext("2d");
    if (
      canvas.width !== video.videoWidth ||
      canvas.height !== video.videoHeight
    ) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    try {
      const timestamp = Math.max(
        performance.now(),
        lastTimestampRef.current + 1
      );
      lastTimestampRef.current = timestamp;

      const nowMs = performance.now();
      if (nowMs - lastDetectionRef.current < 33) {
        animationRef.current = requestAnimationFrame(detectFaceAndHands);
        return;
      }
      lastDetectionRef.current = nowMs;

      const faceResults = faceLandmarker.detectForVideo(video, timestamp);
      const handResults = handLandmarker.detectForVideo(video, timestamp);

      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      let faceBox: {
        minX: number;
        maxX: number;
        minY: number;
        maxY: number;
      } | null = null;

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
        }

        let minX = 1,
          maxX = 0,
          minY = 1,
          maxY = 0;
        landmarks.forEach((pt) => {
          if (pt.x < minX) minX = pt.x;
          if (pt.x > maxX) maxX = pt.x;
          if (pt.y < minY) minY = pt.y;
          if (pt.y > maxY) maxY = pt.y;
        });

        faceBox = { minX, maxX, minY, maxY };

        const leftIris = landmarks[468];
        const rightIris = landmarks[473];

        const leftOuter = landmarks[33];
        const leftInner = landmarks[133];
        const rightInner = landmarks[362];
        const rightOuter = landmarks[263];

        const leftTop = landmarks[159];
        const leftBottom = landmarks[145];
        const rightTop = landmarks[386];
        const rightBottom = landmarks[374];

        if (
          leftIris &&
          rightIris &&
          leftOuter &&
          leftInner &&
          rightInner &&
          rightOuter &&
          leftTop &&
          leftBottom &&
          rightTop &&
          rightBottom
        ) {
          const leftRatioH = horizontalRatio(leftOuter, leftInner, leftIris);
          const rightRatioH = horizontalRatio(rightInner, rightOuter, rightIris);

          const averageHorizontalRatio = (leftRatioH + rightRatioH) / 2;

          const leftHeight = Math.abs(leftBottom.y - leftTop.y);
          const rightHeight = Math.abs(rightBottom.y - rightTop.y);

          const leftRatioV =
            leftHeight > 0.001 ? (leftIris.y - leftTop.y) / leftHeight : 0.5;
          const rightRatioV =
            rightHeight > 0.001 ? (rightIris.y - rightTop.y) / rightHeight : 0.5;

          const averageVerticalRatio = (leftRatioV + rightRatioV) / 2;

          const isLookingSide =
            averageHorizontalRatio < 0.32 || averageHorizontalRatio > 0.68;

          const isLookingUp = averageVerticalRatio < 0.30;
          const isLookingDown = averageVerticalRatio > 0.62;

          const now = Date.now();

          if (isLookingSide) {
            sideGazeFramesRef.current += 1;
          } else {
            sideGazeFramesRef.current = Math.max(0, sideGazeFramesRef.current - 1);
          }

          if (
            sideGazeFramesRef.current >= 5 &&
            now - lastSideDropRef.current > 2500
          ) {
            lastSideDropRef.current = now;
            sideGazeFramesRef.current = 0;
            applyRiskAdjustment(5, "Eye direction violation: Looking to the side");
          }

          if (isLookingUp || isLookingDown) {
            verticalGazeFramesRef.current += 1;
          } else {
            verticalGazeFramesRef.current = Math.max(
              0,
              verticalGazeFramesRef.current - 1
            );
          }

          if (
            verticalGazeFramesRef.current >= 4 &&
            now - lastVerticalDropRef.current > 2500
          ) {
            lastVerticalDropRef.current = now;
            verticalGazeFramesRef.current = 0;
            const directionText = isLookingUp ? "upward" : "downward";
            applyRiskAdjustment(
              5,
              `Eye direction violation: Looking ${directionText}`
            );
          }
        }
      } else {
        const now = Date.now();
        if (noFaceSinceRef.current === null) {
          noFaceSinceRef.current = now;
        }
        if (now - noFaceSinceRef.current > 2000) {
          applyRiskAdjustment(5, "Face lost — attention risk");
          noFaceSinceRef.current = now;
        }
      }

      let isHandCurrentlyOverFace = false;

      if (
        handResults.landmarks &&
        handResults.landmarks.length > 0 &&
        faceBox &&
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
              pt.x >= faceBox!.minX &&
              pt.x <= faceBox!.maxX &&
              pt.y >= faceBox!.minY &&
              pt.y <= faceBox!.maxY
            ) {
              isHandCurrentlyOverFace = true;
            }
          });
        });
      }

      const now = Date.now();

      if (isHandCurrentlyOverFace) {
        if (handTouchStartRef.current === null) {
          handTouchStartRef.current = now;
        }

        const durationTouching = now - handTouchStartRef.current;

        if (durationTouching > 1000 && now - lastHandDropRef.current > 3000) {
          lastHandDropRef.current = now;
          applyRiskAdjustment(5, "Hand covering face/mouth detected");
        }
      } else {
        handTouchStartRef.current = null;
      }
    } catch (error) {
      console.error("MediaPipe processing error:", error);
    }

    animationRef.current = requestAnimationFrame(detectFaceAndHands);
  };

  useEffect(() => {
    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
      stopCameraStream();

      if (faceLandmarkerRef.current) {
        faceLandmarkerRef.current.close();
        faceLandmarkerRef.current = null;
      }
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
        handLandmarkerRef.current = null;
      }
    };
  }, []);

  return (
    <main className="min-h-screen bg-[#050816] text-white">
      <header className="border-b border-white/10 px-8 py-5">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">VeriProxy-AI</h1>
            <p className="text-sm text-gray-400">
              AI-Powered Interview Integrity Monitor
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
              className="h-full w-full object-cover"
            />

            <canvas
              ref={canvasRef}
              className="pointer-events-none absolute inset-0 h-full w-full object-cover"
            />

            {!cameraOn && (
              <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                Camera preview
              </div>
            )}
          </div>

          <button
            onClick={startCamera}
            disabled={cameraOn}
            className="mt-5 w-full rounded-xl bg-blue-600 px-5 py-3 font-semibold transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {cameraOn ? "Camera Active" : "Start Camera"}
          </button>

          <button
            onClick={() =>
              applyRiskAdjustment(
                10,
                "External Audio Flag: Bluetooth / Earbud Detected"
              )
            }
            className="mt-3 w-full rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2 text-xs text-red-400 transition hover:bg-red-500/20"
          >
            ⚠️ Test Bluetooth Detection
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