from __future__ import annotations

import argparse
import asyncio
import contextlib
import fractions
import signal
import threading
import time
from pathlib import Path

import av
import grpc
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from artemis_cve.protos.detector import common_pb2
from artemis_cve.protos.detector import webrtc_detector_pb2 as pb2
from artemis_cve.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_PATH = ROOT / "data-bin" / "pen.mp4"


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
        frame = VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = self._frame_index
        frame.time_base = self._time_base
        self._frame_index += 1
        return frame

    def stop(self) -> None:
        super().stop()
        if self._container is not None:
            self._container.close()
            self._container = None


def _format_detection_reply(det_reply: pb2.StreamDetectionsReply | None) -> str:
    if det_reply is None:
        return "No detections received yet."

    if not det_reply.detections:
        return (
            f"frame_id={det_reply.frame_id} pts_ms={det_reply.pts_ms} "
            "detections=0"
        )

    parts = [
        f"frame_id={det_reply.frame_id} pts_ms={det_reply.pts_ms} "
        f"detections={len(det_reply.detections)}"
    ]
    for index, detection in enumerate(det_reply.detections, start=1):
        if detection.geometry.HasField("box"):
            box = detection.geometry.box
            geom = (
                f"box=({int(box.x_min)}, {int(box.y_min)}, "
                f"{int(box.x_max)}, {int(box.y_max)})"
            )
        else:
            geom = "box=<missing>"
        parts.append(
            f"  {index}. class={detection.class_name} "
            f"score={detection.score:.3f} {geom}"
        )
    return "\n".join(parts)


async def run_client(
    server_addr: str,
    video_path: Path,
    score_threshold: float,
    print_interval: float,
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

    latest_reply: pb2.StreamDetectionsReply | None = None
    det_lock = threading.Lock()

    async def recv_detections() -> None:
        nonlocal latest_reply
        try:
            async for det_reply in stub.StreamDetections(pb2.StreamDetectionsRequest(stream_id=stream_id)):
                with det_lock:
                    latest_reply = det_reply
        finally:
            stop_event.set()

    async def print_detections() -> None:
        while not stop_event.is_set():
            await asyncio.sleep(print_interval)
            with det_lock:
                snapshot = latest_reply
            print(_format_detection_reply(snapshot))

    det_task = asyncio.create_task(recv_detections())
    print_task = asyncio.create_task(print_detections())

    try:
        while not stop_event.is_set():
            if video_track.readyState == "ended":
                stop_event.set()
                break
            await asyncio.sleep(0.1)
    finally:
        for task in (det_task, print_task):
            task.cancel()
        for task in (det_task, print_task):
            with contextlib.suppress(asyncio.CancelledError, grpc.RpcError):
                await task
        await pc.close()
        await channel.close()
        stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="No-GUI WebRTC demo client for artemis-cve.")
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--print-interval", type=float, default=5.0)
    args = parser.parse_args()
    asyncio.run(
        run_client(
            server_addr=args.server,
            video_path=Path(args.video),
            score_threshold=args.score_threshold,
            print_interval=args.print_interval,
        )
    )


if __name__ == "__main__":
    main()
