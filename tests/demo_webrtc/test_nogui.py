from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
from datetime import datetime
import fractions
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
from tqdm import tqdm

from artemis_cve.protos.detector import common_pb2
from artemis_cve.protos.detector import webrtc_detector_pb2 as pb2
from artemis_cve.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc

DEFAULT_VIDEO_PATH = Path("data-bin/1082895552-1-208.mp4")


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
        self._started_at = time.perf_counter()
        self._frame_buffer: collections.OrderedDict[int, object] = collections.OrderedDict()
        self._frame_buffer_lock = threading.Lock()

    async def recv(self) -> VideoFrame:
        try:
            av_frame = next(self._frames)
        except StopIteration as exc:
            self.stop()
            raise MediaStreamError from exc

        bgr = av_frame.to_ndarray(format="bgr24")
        frame = VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = self._frame_index
        frame.time_base = self._time_base
        pts_ms = int(round(float(frame.pts * frame.time_base) * 1000.0))
        with self._frame_buffer_lock:
            self._frame_buffer[pts_ms] = bgr.copy()
            while len(self._frame_buffer) > 300:
                self._frame_buffer.popitem(last=False)
        self._frame_index += 1
        return frame

    @property
    def frame_index(self) -> int:
        """返回当前已发送的帧数。"""

        return self._frame_index

    @property
    def source_fps(self) -> float:
        """返回输入视频声明的原始帧率。"""

        return self._fps

    def current_fps(self) -> float:
        """按实际发送节奏计算当前平均帧率。"""

        elapsed = time.perf_counter() - self._started_at
        if elapsed <= 0:
            return 0.0
        return self._frame_index / elapsed

    def pop_buffered_frame(self, pts_ms: int) -> object | None:
        """按时间戳取出对应的原始视频帧。"""

        with self._frame_buffer_lock:
            return self._frame_buffer.pop(pts_ms, None)

    def stop(self) -> None:
        super().stop()
        if self._container is not None:
            self._container.close()
            self._container = None


class VideoResultWriter:
    """把带检测结果的视频帧写入本地文件。"""

    def __init__(self, output_path: Path, fps: float) -> None:
        self.output_path = output_path
        self.fps = max(1.0, fps)
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: object) -> None:
        """写入一帧 BGR 图像；首次写入时自动初始化 writer。"""

        height, width = frame.shape[:2]
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open output video for writing: {self.output_path}")
        self._writer.write(frame)

    def close(self) -> None:
        """释放底层视频写入器。"""

        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _draw_detections(
    image: object,
    det_reply: pb2.StreamDetectionsReply,
) -> object:
    """在图像上绘制检测框与类别信息。"""

    annotated = image.copy()
    for detection in det_reply.detections:
        if not detection.geometry.HasField("box"):
            continue
        box = detection.geometry.box
        x1, y1, x2, y2 = map(int, [box.x_min, box.y_min, box.x_max, box.y_max])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"{detection.class_name} {detection.score:.3f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return annotated


def _build_output_video_path(
    video_path: Path,
    output_root: Path,
) -> Path:
    """生成 `outputs/<时间戳>/` 下的视频输出路径。"""

    timestamp_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_root / timestamp_dir / f"{video_path.stem}-annotated.mp4"


async def run_client(
    server_addr: str,
    video_path: Path,
    score_threshold: float,
    output_video: Path,
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

    latest_reply: pb2.StreamDetectionsReply | None = None
    det_lock = threading.Lock()
    result_writer = VideoResultWriter(output_video, fps=video_track.source_fps)

    async def recv_detections() -> None:
        nonlocal latest_reply
        try:
            async for det_reply in stub.StreamDetections(pb2.StreamDetectionsRequest(stream_id=stream_id)):
                with det_lock:
                    latest_reply = det_reply
                frame = video_track.pop_buffered_frame(det_reply.pts_ms)
                if frame is not None:
                    annotated = _draw_detections(frame, det_reply)
                    result_writer.write(annotated)
        finally:
            stop_event.set()

    det_task = asyncio.create_task(recv_detections())
    progress = tqdm(
        total=None,
        desc=f"stream {stream_id}",
        unit="frame",
        dynamic_ncols=True,
    )
    last_frame_index = 0

    try:
        while not stop_event.is_set():
            current_frame_index = video_track.frame_index
            if current_frame_index > last_frame_index:
                progress.update(current_frame_index - last_frame_index)
                last_frame_index = current_frame_index

            with det_lock:
                snapshot = latest_reply
            progress.set_postfix(
                fps=f"{video_track.current_fps():.2f}",
                detections=0 if snapshot is None else len(snapshot.detections),
                refresh=False,
            )

            if video_track.readyState == "ended":
                stop_event.set()
                break
            await asyncio.sleep(0.1)
    finally:
        if video_track.frame_index > last_frame_index:
            progress.update(video_track.frame_index - last_frame_index)
        progress.set_postfix(fps=f"{video_track.current_fps():.2f}", refresh=True)
        progress.close()
        det_task.cancel()
        for task in (det_task,):
            with contextlib.suppress(asyncio.CancelledError, grpc.RpcError):
                await task
        result_writer.close()
        await pc.close()
        await channel.close()
        stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="No-GUI WebRTC demo client for artemis-cve.")
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()
    video_path = Path(args.video)
    output_video = _build_output_video_path(video_path, Path(args.output_root))
    asyncio.run(
        run_client(
            server_addr=args.server,
            video_path=video_path,
            score_threshold=args.score_threshold,
            output_video=output_video,
        )
    )


if __name__ == "__main__":
    main()
