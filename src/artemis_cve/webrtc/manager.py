from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence

from artemis_cve.inferencers.yolo import SharedYoloBoxInferencer
from artemis_cve.protos.detector import common_pb2

from .session import WebRtcSession

logger = logging.getLogger(__name__)


class WebRtcSessionManager:
    def __init__(
        self,
        model_dir: str,
        textencoder_model_dir: str,
        class_names: Sequence[str],
        device: str,
        dtype: str,
        use_cuda_graph: bool = False,
    ) -> None:
        self._sessions: dict[str, WebRtcSession] = {}
        self.inferencer = SharedYoloBoxInferencer(
            model_dir=model_dir,
            textencoder_model_dir=textencoder_model_dir,
            class_names=class_names,
            device=device,
            dtype=dtype,
            use_cuda_graph=use_cuda_graph,
        )

    def create(self, config: common_pb2.StreamConfig | None = None) -> WebRtcSession:
        stream_id = str(uuid.uuid4())
        score_threshold = float(config.score_threshold) if config else 0.0
        max_detections = int(config.max_detections) if config and config.max_detections > 0 else None
        session = WebRtcSession(
            stream_id=stream_id,
            inferencer=self.inferencer,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        self._sessions[stream_id] = session
        logger.info("Session created: %s", stream_id)
        return session

    def get(self, stream_id: str) -> WebRtcSession | None:
        return self._sessions.get(stream_id)

    async def remove(self, stream_id: str) -> None:
        session = self._sessions.pop(stream_id, None)
        if session is not None:
            await session.close()
