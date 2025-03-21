import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";
import axios from "axios";
import {
  Upload,
  Camera,
  RefreshCw,
  CheckCircle2,
  XCircle,
  User,
  AlertTriangle,
} from "lucide-react";

const RECOGNITION_INTERVAL = 100;
const MATCH_THRESHOLD = 0.5;
const BOX_COLORS = {
  match: "#22c55e",
  noMatch: "#ef4444",
  unknown: "#eab308",
} as const;

type UnknownFace = {
  timestamp: number;
  notified: boolean;
};

function App() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recognitionInterval = useRef<number>();
  const unknownFacesRef = useRef<Map<string, UnknownFace>>(new Map());

  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [referenceImage, setReferenceImage] = useState<string | null>(null);
  const [referenceFaceDescriptor, setReferenceFaceDescriptor] =
    useState<Float32Array | null>(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>("");
  const [stats, setStats] = useState({
    matchCount: 0,
    totalScans: 0,
    unknownCount: 0,
  });
  const [userName, setUserName] = useState("");
  const [isNameValid, setIsNameValid] = useState(false);

  const accuracy = useMemo(() => {
    return stats.totalScans > 0
      ? ((stats.matchCount / stats.totalScans) * 100).toFixed(1)
      : "0.0";
  }, [stats.matchCount, stats.totalScans]);

  const loadCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );
      setCameras(videoDevices);
      if (videoDevices.length > 0) {
        setSelectedCamera(videoDevices[0].deviceId);
      }
    } catch (error) {
      console.error("Error loading cameras:", error);
    }
  }, []);

  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
          faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
          faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
        ]);
        setIsModelLoaded(true);
      } catch (error) {
        console.error("Error loading models:", error);
      }
    };

    loadModels();
    loadCameras();

    return () => {
      if (recognitionInterval.current) {
        clearInterval(recognitionInterval.current);
      }
    };
  }, [loadCameras]);

  const notifyUnknownFace = () => {
    axios
      .post("http://localhost:3000/call-owner", {
        phone: "+8801729414662",
      })
      .then((res) => console.log(res.data))
      .catch((err) => console.error(err));
  };

  class FalseTracker {
    private falseCount: number;
    private threshold: number;
    private delay: number;
    private timer: NodeJS.Timeout | null;
    private alerted: boolean;

    constructor(threshold: number = 5, delay: number = 60000) {
      this.falseCount = 0;
      this.threshold = threshold;
      this.delay = delay;
      this.timer = null;
      this.alerted = false;
    }

    public check(value: boolean): void {
      if (value) {
        this.falseCount = 0; // Reset counter if true appears
        this.alerted = false;
        if (this.timer) {
          clearTimeout(this.timer);
          this.timer = null;
        }
      } else {
        this.falseCount++;
        if (this.falseCount >= this.threshold && !this.alerted) {
          // console.log("Warning: Multiple false values detected!");
          notifyUnknownFace();
          this.alerted = true;

          // If continues to be false, trigger again after delay
          this.timer = setTimeout(() => {
            this.alerted = false;
          }, this.delay);
        }
      }
    }
  }

  // Usage Example
  const tracker = new FalseTracker(5, 60000);

  const handleImageUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file || !isNameValid) return;

      const imageUrl = URL.createObjectURL(file);
      setReferenceImage(imageUrl);

      try {
        const img = await faceapi.fetchImage(imageUrl);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detections) {
          setReferenceFaceDescriptor(detections.descriptor);
          setStats({ matchCount: 0, totalScans: 0, unknownCount: 0 });
          unknownFacesRef.current.clear();
        }
      } catch (error) {
        console.error("Error processing reference image:", error);
      }
    },
    [isNameValid]
  );

  const processFrame = useCallback(async () => {
    if (
      !webcamRef.current?.video ||
      !canvasRef.current ||
      !referenceFaceDescriptor
    )
      return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;

    try {
      const detections = await faceapi
        .detectAllFaces(video)
        .withFaceLandmarks()
        .withFaceDescriptors();

      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight,
      };
      faceapi.matchDimensions(canvas, displaySize);

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      const context = canvas.getContext("2d");

      if (context) {
        context.clearRect(0, 0, canvas.width, canvas.height);

        let matchesThisFrame = 0;
        let unknownThisFrame = 0;

        resizedDetections.forEach((detection, index) => {
          const distance = faceapi.euclideanDistance(
            referenceFaceDescriptor,
            detection.descriptor
          );
          const isMatch = distance < MATCH_THRESHOLD;

          if (isMatch) {
            matchesThisFrame++;
            setTimeout(() => {
              tracker.check(true);
            }, index * 60000);
          } else {
            // Handle unknown face
            const faceId = detection.descriptor.toString();
            const now = Date.now();
            const unknownFace = unknownFacesRef.current.get(faceId);

            if (!unknownFace) {
              unknownFacesRef.current.set(faceId, {
                timestamp: now,
                notified: false,
              });
              unknownThisFrame++;
              setTimeout(() => {
                tracker.check(false);
              }, index * 1000);
            }
            //  else if (
            //   !unknownFace.notified &&
            //   now - unknownFace.timestamp > 5000
            // ) {
            //   notifyUnknownFace();
            //   unknownFacesRef.current.set(faceId, {
            //     timestamp: now,
            //     notified: true,
            //   });
            // }
          }

          new faceapi.draw.DrawBox(detection.detection.box, {
            label: isMatch ? `Match: ${userName}` : "Unknown Person",
            boxColor: isMatch ? BOX_COLORS.match : BOX_COLORS.unknown,
            lineWidth: 2,
            drawLabelOptions: {
              fontSize: 16,
              fontStyle: "bold",
              padding: 8,
            },
          }).draw(canvas);
        });

        if (resizedDetections.length > 0) {
          setStats((prev) => ({
            matchCount: prev.matchCount + matchesThisFrame,
            totalScans: prev.totalScans + resizedDetections.length,
            unknownCount: prev.unknownCount + unknownThisFrame,
          }));
        }
      }
    } catch (error) {
      console.error("Error processing frame:", error);
    }
  }, [referenceFaceDescriptor, userName]);

  const startFaceRecognition = useCallback(() => {
    if (!referenceFaceDescriptor || !isNameValid) return;

    setIsRecognizing(true);
    setStats({ matchCount: 0, totalScans: 0, unknownCount: 0 });
    unknownFacesRef.current.clear();

    recognitionInterval.current = window.setInterval(
      processFrame,
      RECOGNITION_INTERVAL
    );

    return () => {
      if (recognitionInterval.current) {
        clearInterval(recognitionInterval.current);
      }
    };
  }, [referenceFaceDescriptor, processFrame, isNameValid]);

  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const name = event.target.value.trim();
    setUserName(name);
    setIsNameValid(name.length >= 2);
  };

  const cameraOptions = useMemo(() => {
    return cameras.map((camera) => (
      <option key={camera.deviceId} value={camera.deviceId}>
        {camera.label || `Camera ${cameras.indexOf(camera) + 1}`}
      </option>
    ));
  }, [cameras]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-8">
      <div className="container mx-auto">
        <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl shadow-2xl p-8 mb-8">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-4xl font-bold text-white">
              Face Recognition System
            </h1>
            <div className="flex items-center space-x-2">
              <select
                value={selectedCamera}
                onChange={(e) => setSelectedCamera(e.target.value)}
                className="bg-gray-700 text-white rounded-lg px-4 py-2 outline-none focus:ring-2 focus:ring-blue-500"
              >
                {cameraOptions}
              </select>
              <button
                onClick={loadCameras}
                className="p-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
              >
                <RefreshCw className="w-5 h-5" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            <div className="md:col-span-2">
              <div className="relative rounded-xl overflow-hidden bg-gray-800">
                <Webcam
                  ref={webcamRef}
                  className="w-full rounded-xl"
                  // mirrored
                  videoConstraints={{ deviceId: selectedCamera }}
                />
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full"
                />
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6">
              <div className="mb-6">
                <h2 className="text-xl font-semibold text-white mb-4">
                  User Information
                </h2>
                <div className="mb-4">
                  <label className="block text-gray-300 mb-2">Your Name</label>
                  <div className="relative">
                    <input
                      type="text"
                      value={userName}
                      onChange={handleNameChange}
                      placeholder="Enter your name"
                      className="w-full bg-gray-700 text-white rounded-lg px-4 py-2 pl-10 outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <User className="absolute left-3 top-2.5 w-5 h-5 text-gray-400" />
                  </div>
                </div>

                <label className="flex items-center justify-center w-full h-40 border-2 border-dashed border-gray-600 rounded-lg cursor-pointer hover:border-blue-500 transition">
                  {referenceImage ? (
                    <img
                      src={referenceImage}
                      alt="Reference"
                      className="w-full h-full object-cover rounded-lg"
                    />
                  ) : (
                    <div className="text-center">
                      <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                      <span className="text-gray-400">Upload Photo</span>
                    </div>
                  )}
                  <input
                    type="file"
                    className="hidden"
                    accept="image/*"
                    onChange={handleImageUpload}
                    disabled={!isNameValid}
                  />
                </label>
              </div>

              <div className="space-y-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between text-white">
                    <span>Recognition Accuracy</span>
                    <span className="font-semibold">{accuracy}%</span>
                  </div>
                  <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{ width: `${accuracy}%` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center space-x-2 text-green-400">
                      <CheckCircle2 className="w-5 h-5" />
                      <span>Matches</span>
                    </div>
                    <div className="text-2xl font-bold text-white mt-1">
                      {stats.matchCount}
                    </div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center space-x-2 text-yellow-400">
                      <AlertTriangle className="w-5 h-5" />
                      <span>Unknown</span>
                    </div>
                    <div className="text-2xl font-bold text-white mt-1">
                      {stats.unknownCount}
                    </div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center space-x-2 text-red-400">
                      <XCircle className="w-5 h-5" />
                      <span>No Match</span>
                    </div>
                    <div className="text-2xl font-bold text-white mt-1">
                      {stats.totalScans - stats.matchCount - stats.unknownCount}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-center">
            <button
              onClick={startFaceRecognition}
              disabled={
                !isModelLoaded ||
                !referenceFaceDescriptor ||
                isRecognizing ||
                !isNameValid
              }
              className={`flex items-center px-8 py-4 rounded-xl text-white font-semibold text-lg
                ${
                  isModelLoaded &&
                  referenceFaceDescriptor &&
                  !isRecognizing &&
                  isNameValid
                    ? "bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800"
                    : "bg-gray-700 cursor-not-allowed"
                } transition-all duration-300 shadow-lg`}
            >
              <Camera className="w-6 h-6 mr-3" />
              {!isModelLoaded
                ? "Loading Models..."
                : !isNameValid
                ? "Enter Your Name"
                : isRecognizing
                ? "Recognition Active"
                : "Start Recognition"}
            </button>
          </div>
        </div>
      </div>
      <button onClick={() => notifyUnknownFace()}>test</button>
    </div>
  );
}

export default App;
