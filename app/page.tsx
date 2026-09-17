"use client";

import { useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);

  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationRef = useRef<number | null>(null);

  const previousGazeRef = useRef<number | null>(null);
  const lastDropRef = useRef<number>(0);
  const noFaceSinceRef = useRef<number | null>(null);
  const lastTimestampRef = useRef<number>(0);
  const lastDetectionRef = useRef<number>(0);

  const [cameraOn, setCameraOn] = useState(false);
  const [score, setScore] = useState(100);

  const [analysisStatus, setAnalysisStatus] = useState(
    "Waiting for behavioral analysis..."
  );

  const statusRef = useRef("Waiting for behavioral analysis...");

  // Filter out MediaPipe non-fatal INFO log from Next.js error overlay
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

  // -----------------------------------------
  // UPDATE SCORE
  // -----------------------------------------

  const applyRiskAdjustment = (
    change: number,
    message: string
  ) => {
    setScore((currentScore) =>
      Math.max(
        0,
        Math.min(100, currentScore - change)
      )
    );

    statusRef.current = message;
    setAnalysisStatus(message);
  };

  // -----------------------------------------
  // STOP CAMERA
  // -----------------------------------------

  const stopCameraStream = () => {
    const video = videoRef.current;

    if (!video) return;

    const stream = video.srcObject as MediaStream | null;

    if (stream) {
      stream.getTracks().forEach((track) => {
        track.stop();
      });
    }

    video.pause();
    video.srcObject = null;
  };

  // -----------------------------------------
  // START CAMERA
  // -----------------------------------------

  const startCamera = async () => {
    // Stop previous stream if one exists
    stopCameraStream();

    // Stop previous animation
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    // Close previous MediaPipe instance
    if (faceLandmarkerRef.current) {
      faceLandmarkerRef.current.close();
      faceLandmarkerRef.current = null;
    }

    try {
      setAnalysisStatus("Starting camera...");

      // -----------------------------------------
      // CHECK CAMERA SUPPORT
      // -----------------------------------------

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Camera access is not supported in this browser."
        );
      }

      // -----------------------------------------
      // GET CAMERA
      // -----------------------------------------

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: {
              ideal: 1280,
            },
            height: {
              ideal: 720,
            },
            facingMode: "user",
          },
          audio: false,
        });

      const video = videoRef.current;

      if (!video) {
        stream
          .getTracks()
          .forEach((track) => track.stop());

        throw new Error(
          "Video element was not found."
        );
      }

      // Attach camera
      video.srcObject = stream;

      // -----------------------------------------
      // WAIT FOR VIDEO METADATA
      // -----------------------------------------

      await new Promise<void>(
        (resolve, reject) => {
          const timeout = window.setTimeout(() => {
            reject(
              new Error(
                "Camera video did not become ready."
              )
            );
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
        }
      );

      // -----------------------------------------
      // PLAY VIDEO
      // -----------------------------------------

      await video.play();

      // Wait one animation frame so the browser
      // actually has a video frame available.
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });

      setCameraOn(true);
      setAnalysisStatus(
        "Initializing MediaPipe..."
      );

      // -----------------------------------------
      // LOAD MEDIAPIPE
      // -----------------------------------------

      const vision =
        await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
        );

      // -----------------------------------------
      // CREATE FACE LANDMARKER
      // -----------------------------------------

      const faceLandmarker =
        await FaceLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath:
                "/models/face_landmarker.task",
            },

            runningMode: "VIDEO",

            numFaces: 1,

            minFaceDetectionConfidence: 0.5,

            minFacePresenceConfidence: 0.5,

            minTrackingConfidence: 0.5,
          }
        );

      // Save MediaPipe instance
      faceLandmarkerRef.current =
        faceLandmarker;

      // Reset analysis values
      previousGazeRef.current = null;
      lastDropRef.current = 0;
      noFaceSinceRef.current = null;
      lastTimestampRef.current = 0;
      lastDetectionRef.current = 0;

      statusRef.current =
        "Behavioral analysis active";

      setAnalysisStatus(
        "Behavioral analysis active"
      );

      // Start detection loop
      startDetection();
    } catch (error) {
      console.error(
        "Camera / MediaPipe error:",
        error
      );

      setCameraOn(false);

      const isCameraBusy =
        error instanceof DOMException &&
        error.name === "NotReadableError";

      const isPermissionDenied =
        error instanceof DOMException &&
        error.name === "NotAllowedError";

      let message =
        "Unable to start analysis.";

      if (isCameraBusy) {
        message =
          "Camera is already in use. Close other apps using the camera and try again.";
      } else if (isPermissionDenied) {
        message =
          "Camera permission was denied. Allow camera access and try again.";
      } else if (error instanceof Error) {
        message = error.message;
      }

      setAnalysisStatus(message);

      alert(message);
    }
  };

  // -----------------------------------------
  // START DETECTION LOOP
  // -----------------------------------------

  const startDetection = () => {
    if (animationRef.current !== null) {
      cancelAnimationFrame(
        animationRef.current
      );
    }

    animationRef.current =
      requestAnimationFrame(detectFace);
  };

  // -----------------------------------------
  // FACE DETECTION
  // -----------------------------------------

  const detectFace = () => {
    const video = videoRef.current;
    const faceLandmarker = faceLandmarkerRef.current;

    if (!video || !faceLandmarker) {
      animationRef.current = requestAnimationFrame(detectFace);
      return;
    }

    if (
      video.readyState < 2 ||
      video.videoWidth <= 0 ||
      video.videoHeight <= 0 ||
      video.paused ||
      video.ended
    ) {
      animationRef.current = requestAnimationFrame(detectFace);
      return;
    }

    try {
      const timestamp = Math.max(
        performance.now(),
        lastTimestampRef.current + 1
      );
      lastTimestampRef.current = timestamp;

      const nowMs = performance.now();
      if (nowMs - lastDetectionRef.current < 33) {
        animationRef.current = requestAnimationFrame(detectFace);
        return;
      }
      lastDetectionRef.current = nowMs;

      const results = faceLandmarker.detectForVideo(video, timestamp);

      // -----------------------------------------
      // FACE FOUND
      // -----------------------------------------
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        noFaceSinceRef.current = null;

        // Landmarks for Eyes & Iris
        const leftIris = landmarks[468];
        const rightIris = landmarks[473];

        const leftOuter = landmarks[33];
        const leftInner = landmarks[133];
        const rightInner = landmarks[362];
        const rightOuter = landmarks[263];

        if (
          leftIris &&
          rightIris &&
          leftOuter &&
          leftInner &&
          rightInner &&
          rightOuter
        ) {
          const leftWidth = leftOuter.x - leftInner.x;
          const rightWidth = rightOuter.x - rightInner.x;

          const leftRatio =
            Math.abs(leftWidth) > 0.001
              ? (leftIris.x - leftInner.x) / leftWidth
              : 0.5;

          const rightRatio =
            Math.abs(rightWidth) > 0.001
              ? (rightIris.x - rightInner.x) / rightWidth
              : 0.5;

          const currentGazeX = (leftRatio + rightRatio) / 2;

          const previousGaze = previousGazeRef.current;

          if (previousGaze !== null) {
            // Lowered threshold to 0.04 to detect normal eye turns
            const movementX = Math.abs(currentGazeX - previousGaze);

            const now = Date.now();

            if (movementX > 0.04 && now - lastDropRef.current > 1200) {
              lastDropRef.current = now;
              applyRiskAdjustment(5, "Horizontal gaze movement detected");
            }
          }

          previousGazeRef.current = currentGazeX;
        }

        if (statusRef.current !== "Face detected — monitoring gaze") {
          statusRef.current = "Face detected — monitoring gaze";
          setAnalysisStatus("Face detected — monitoring gaze");
        }
      }

      // -----------------------------------------
      // FACE NOT FOUND
      // -----------------------------------------
      else {
        const now = Date.now();

        if (noFaceSinceRef.current === null) {
          noFaceSinceRef.current = now;
        }

        const timeWithoutFace = now - noFaceSinceRef.current;

        if (timeWithoutFace > 2000) {
          applyRiskAdjustment(8, "Face lost — attention risk");
          noFaceSinceRef.current = now;
        }

        if (statusRef.current !== "Face not detected") {
          statusRef.current = "Face not detected";
          setAnalysisStatus("Face not detected");
        }
      }
    } catch (error) {
      console.error("MediaPipe frame error:", error);
    }

    animationRef.current = requestAnimationFrame(detectFace);
  };

  // -----------------------------------------
  // CLEANUP
  // -----------------------------------------

  useEffect(() => {
    return () => {
      if (
        animationRef.current !==
        null
      ) {
        cancelAnimationFrame(
          animationRef.current
        );
      }

      stopCameraStream();

      if (
        faceLandmarkerRef.current
      ) {
        faceLandmarkerRef.current.close();

        faceLandmarkerRef.current =
          null;
      }
    };
  }, []);

  // -----------------------------------------
  // UI
  // -----------------------------------------

  return (
    <main className="min-h-screen bg-[#050816] text-white">

      {/* HEADER */}

      <header className="border-b border-white/10 px-8 py-5">

        <div className="mx-auto flex max-w-6xl items-center justify-between">

          <div>

            <h1 className="text-2xl font-bold">
              VeriProxy-AI
            </h1>

            <p className="text-sm text-gray-400">
              AI-Powered Interview Integrity Monitor
            </p>

          </div>

          <div className="rounded-full border border-green-500/30 bg-green-500/10 px-4 py-2 text-sm text-green-400">

            ● System Ready

          </div>

        </div>

      </header>

      {/* MAIN */}

      <section className="mx-auto grid max-w-6xl gap-8 px-8 py-10 md:grid-cols-2">

        {/* CAMERA PANEL */}

        <div className="rounded-2xl border border-white/10 bg-white/5 p-5">

          <div className="mb-4 flex items-center justify-between">

            <h2 className="text-lg font-semibold">
              Candidate Camera
            </h2>

            <span className="text-sm text-gray-400">

              {cameraOn
                ? "Camera Active"
                : "Camera Off"}

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

            {cameraOn
              ? "Camera Active"
              : "Start Camera"}

          </button>

        </div>

        {/* SCORE PANEL */}

        <div className="rounded-2xl border border-white/10 bg-white/5 p-5">

          <h2 className="text-lg font-semibold">
            Integrity Score
          </h2>

          <div className="mt-10 flex flex-col items-center">

            <div className="flex h-48 w-48 items-center justify-center rounded-full border-8 border-blue-500/30">

              <div className="text-center">

                <div className="text-6xl font-bold">
                  {score}
                </div>

                <div className="mt-2 text-sm text-gray-400">
                  Risk Score
                </div>

              </div>

            </div>

            <p className="mt-8 text-center text-gray-400">

              {analysisStatus}

            </p>

          </div>

        </div>

      </section>

    </main>
  );
}