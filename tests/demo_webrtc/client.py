from __future__ import annotations

import argparse
import asyncio
import contextlib
import collections
import fractions
import queue
import signal
import threading
import time
from pathlib import Path

import av
import cv2
import grpc
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from artemis_cve.protos.detector import common_pb2
from artemis_cve.protos.detector import webrtc_detector_pb2 as pb2
from artemis_cve.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc

DEFAULT_VIDEO_PATH = Path("data-bin/car.mp4")


class VideoFileTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, video_path: Path):
        super().__init__()
        self._container = av.open(str(video_path))
        self._frames = self._container.decode(video=0)
        self._stream = self._container.streams.video[0]
        self._fps = float(self._stream.average_rate or 30)
        self._time_base = fractions.Fraction(1, max(1, round(self._fps)))
        self._frame_index = 0
        self.display_queue: queue.Queue = queue.Queue(maxsize=4)
        self._sent_timestamps_ms: collections.OrderedDict[int, float] = collections.OrderedDict()
        self._sent_timestamps_lock = threading.Lock()
        self._start = time.time()

    async def recv(self) -> VideoFrame:
        try:
            av_frame = next(self._frames)
        except StopIteration as exc:
            self.stop()
            raise MediaStreamError from exc

        target = self._start + self._frame_index / self._fps
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        bgr = av_frame.to_ndarray(format="bgr24")
        try:
            self.display_queue.put_nowait(bgr.copy())
        except queue.Full:
            pass

        frame = VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = self._frame_index
        frame.time_base = self._time_base
        pts_ms = int(round(float(frame.pts * frame.time_base) * 1000.0))
        with self._sent_timestamps_lock:
            self._sent_timestamps_ms[pts_ms] = time.perf_counter()
            while len(self._sent_timestamps_ms) > 300:
                self._sent_timestamps_ms.popitem(last=False)
        self._frame_index += 1
        return frame

    def pop_sent_timestamp(self, pts_ms: int) -> float | None:
        with self._sent_timestamps_lock:
            return self._sent_timestamps_ms.pop(pts_ms, None)

    def stop(self) -> None:
        super().stop()
        if self._container is not None:
            self._container.close()
            self._container = None


async def run_client(
    server_addr: str,
    video_path: Path,
    score_threshold: float,
) -> None:
    channel = grpc.aio.insecure_channel(server_addr)
    stub = pb2_grpc.WebRtcDetectorEngineStub(channel)

    reply = await stub.CreateStream(
        pb2.CreateStreamRequest(
            config=common_pb2.StreamConfig(
                video_codec="vp8",
                score_threshold=score_threshold,
            )
        )
    )
    stream_id = reply.stream_id
    print(f"Stream created: {stream_id}")

    pc = RTCPeerConnection()
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop_event.set)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            stop_event.set()

    video_track = VideoFileTrack(video_path)
    pc.addTrack(video_track)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=reply.offer.sdp, type=reply.offer.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    await stub.UpdateStream(
        pb2.StreamSignal(
            stream_id=stream_id,
            answer=pb2.SessionDescription(
                type=pc.localDescription.type,
                sdp=pc.localDescription.sdp,
            ),
        )
    )
    print("WebRTC connected")

    latest_reply = None
    det_lock = threading.Lock()
    display_fps_text = "Display FPS: --.-"
    display_counter = 0
    display_window_start = time.perf_counter()
    detection_fps_text = "Detection FPS: --.-"
    detection_counter = 0
    detection_window_start = time.perf_counter()
    latency_text = "Latency: --.- ms"
    latency_print_window_start = time.perf_counter()

    async def recv_detections() -> None:
        nonlocal detection_counter, detection_fps_text, detection_window_start, latest_reply
        nonlocal latency_print_window_start, latency_text
        try:
            async for det_reply in stub.StreamDetections(pb2.StreamDetectionsRequest(stream_id=stream_id)):
                detection_counter += 1
                now = time.perf_counter()
                elapsed = now - detection_window_start
                if elapsed >= 1.0:
                    detection_fps_text = f"Detection FPS: {detection_counter / elapsed:.1f}"
                    detection_counter = 0
                    detection_window_start = now

                sent_at = video_track.pop_sent_timestamp(det_reply.pts_ms)
                if sent_at is not None:
                    latency_ms = (now - sent_at) * 1000.0
                    latency_text = f"Latency: {latency_ms:.1f} ms"
                    if now - latency_print_window_start >= 1.0:
                        print(
                            f"[latency] frame_id={det_reply.frame_id} "
                            f"pts_ms={det_reply.pts_ms} latency_ms={latency_ms:.1f}"
                        )
                        latency_print_window_start = now
                with det_lock:
                    latest_reply = det_reply
        finally:
            stop_event.set()

    det_task = asyncio.create_task(recv_detections())

    try:
        while not stop_event.is_set():
            try:
                image = video_track.display_queue.get_nowait()
            except queue.Empty:
                image = None

            if image is not None:
                display_counter += 1
                now = time.perf_counter()
                elapsed = now - display_window_start
                if elapsed >= 1.0:
                    display_fps_text = f"Display FPS: {display_counter / elapsed:.1f}"
                    display_counter = 0
                    display_window_start = now

                with det_lock:
                    det_reply = latest_reply

                if det_reply and det_reply.detections:
                    for detection in det_reply.detections:
                        if not detection.geometry.HasField("box"):
                            continue
                        box = detection.geometry.box
                        x1, y1, x2, y2 = map(int, [box.x_min, box.y_min, box.x_max, box.y_max])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            image,
                            f"{detection.class_name} {detection.score:.3f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

                cv2.putText(
                    image,
                    display_fps_text,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    detection_fps_text,
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    image,
                    latency_text,
                    (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("artemis-cve demo", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            if video_track.readyState == "ended" and video_track.display_queue.empty():
                stop_event.set()
                break
            await asyncio.sleep(0.001 if image is None else 0)
    finally:
        det_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, grpc.RpcError):
            await det_task
        await pc.close()
        await channel.close()
        stop_event.set()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo WebRTC client for artemis-cve.")
    parser.add_argument("--server", default="192.168.1.24:50051")
    parser.add_argument("--video", default=str("data-bin/1082895552-1-208.mp4"))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    args = parser.parse_args()
    asyncio.run(
        run_client(
            server_addr=args.server,
            video_path=Path(args.video),
            score_threshold=args.score_threshold,
        )
    )


if __name__ == "__main__":
    main()
