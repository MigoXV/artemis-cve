from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence

import grpc
import grpc.aio

from artemis_cve.inferencers.yolo import BoxDetection
from artemis_cve.protos.detector import common_pb2
from artemis_cve.protos.detector import webrtc_detector_pb2 as pb2
from artemis_cve.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc
from artemis_cve.webrtc import WebRtcSessionManager

logger = logging.getLogger(__name__)


class WebRtcDetectorServicer(pb2_grpc.WebRtcDetectorEngineServicer):
    def __init__(
        self,
        model_dir: str,
        textencoder_model_dir: str,
        class_names: Sequence[str],
        device: str,
        dtype: str,
        use_cuda_graph: bool = False,
    ) -> None:
        self._manager = WebRtcSessionManager(
            model_dir=model_dir,
            textencoder_model_dir=textencoder_model_dir,
            class_names=class_names,
            device=device,
            dtype=dtype,
            use_cuda_graph=use_cuda_graph,
        )

    @staticmethod
    def _build_detection_proto(detection: BoxDetection) -> common_pb2.Detection:
        x1, y1, x2, y2 = detection.pixel_xyxy
        nx1, ny1, nx2, ny2 = detection.normalized_xyxy
        return common_pb2.Detection(
            class_name=detection.class_name,
            class_id=detection.class_id,
            score=float(detection.score),
            geometry=common_pb2.DetectionGeometry(
                box=common_pb2.BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
            ),
            normalized_geometry=common_pb2.DetectionGeometry(
                box=common_pb2.BoundingBox(
                    x_min=nx1,
                    y_min=ny1,
                    x_max=nx2,
                    y_max=ny2,
                )
            ),
        )

    @classmethod
    def _build_stream_detections_reply(
        cls,
        stream_id: str,
        request_id: str,
        frame_id: int,
        pts_ms: int,
        detections: Sequence[BoxDetection],
    ) -> pb2.StreamDetectionsReply:
        return pb2.StreamDetectionsReply(
            stream_id=stream_id,
            request_id=request_id,
            pts_ms=pts_ms,
            detections=[cls._build_detection_proto(item) for item in detections],
            frame_id=frame_id,
        )

    async def CreateStream(
        self,
        request: pb2.CreateStreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CreateStreamReply:
        session = self._manager.create(request.config)
        local_desc = await session.create_offer()
        return pb2.CreateStreamReply(
            stream_id=session.stream_id,
            offer=pb2.SessionDescription(type=local_desc.type, sdp=local_desc.sdp),
        )

    async def UpdateStream(
        self,
        request: pb2.StreamSignal,
        context: grpc.aio.ServicerContext,
    ) -> pb2.UpdateStreamReply:
        session = self._manager.get(request.stream_id)
        if session is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Stream not found")

        signal_kind = request.WhichOneof("signal")
        if signal_kind == "answer":
            await session.set_answer(request.answer.sdp, request.answer.type)
        elif signal_kind == "ice_candidate":
            logger.debug("Ignoring ICE candidate for stream %s", request.stream_id)
        else:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing stream signal")

        return pb2.UpdateStreamReply()

    async def StreamDetections(
        self,
        request: pb2.StreamDetectionsRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb2.StreamDetectionsReply]:
        session = self._manager.get(request.stream_id)
        if session is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Stream not found")

        det_queue = session.attach_detection_queue()
        try:
            while session.running:
                try:
                    stream_id, request_id, frame_id, pts_ms, detections = await asyncio.wait_for(
                        det_queue.get(),
                        timeout=1.0,
                    )
                    yield self._build_stream_detections_reply(
                        stream_id=stream_id,
                        request_id=request_id,
                        frame_id=frame_id,
                        pts_ms=pts_ms,
                        detections=detections,
                    )
                except asyncio.TimeoutError:
                    continue
        except Exception:
            logger.exception("StreamDetections failed for stream %s", request.stream_id)
            raise
        finally:
            session.detach_detection_queue(det_queue)
            if not session.running and not session.detection_queues:
                await self._manager.remove(request.stream_id)
