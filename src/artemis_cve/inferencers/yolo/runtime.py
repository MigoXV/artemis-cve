from __future__ import annotations

from dataclasses import dataclass

import torch
from ultralytics.utils import nms
from ultralytics.utils.tal import make_anchors

from artemis_cve.models.yolo26e import YOLOEModel, YOLOERawOutput


@dataclass
class RawYoloOutput:
    boxes: list[torch.Tensor] | None = None
    scores: list[torch.Tensor] | None = None
    labels: list[torch.Tensor] | None = None
    masks: list[torch.Tensor | None] | None = None
    raw_results: YOLOERawOutput | torch.Tensor | None = None


@dataclass(slots=True)
class _CUDAGraphState:
    graph: torch.cuda.CUDAGraph
    static_input: torch.Tensor
    static_text_embeddings: torch.Tensor
    static_output: YOLOERawOutput


class BaseYoloRuntime:
    def __init__(self, model: YOLOEModel) -> None:
        self.model = model
        self.config = model.config

    def _validate_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"pixel_values must be a torch.Tensor, got {type(pixel_values)!r}")
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape [batch, channels, height, width]. "
                f"Received shape: {tuple(pixel_values.shape)}"
            )
        if pixel_values.shape[1] != self.config.num_channels:
            raise ValueError(
                f"Expected {self.config.num_channels} channels, received {pixel_values.shape[1]}."
            )
        if not pixel_values.is_floating_point():
            pixel_values = pixel_values.float()
        return pixel_values

    def _validate_text_embeddings(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        if not isinstance(text_embeddings, torch.Tensor):
            raise TypeError(f"text_embeddings must be a torch.Tensor, got {type(text_embeddings)!r}")
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        if text_embeddings.ndim != 3:
            raise ValueError(
                "text_embeddings must have shape [batch, classes, embedding_dim] or [classes, embedding_dim]. "
                f"Received shape: {tuple(text_embeddings.shape)}"
            )
        return text_embeddings

    def _empty_output(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        return (
            torch.empty((0, 4), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            None,
        )

    def _generate_anchor_tensors(
        self,
        raw_output: YOLOERawOutput,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw_output.feature_shapes is None or raw_output.strides is None:
            raise ValueError("raw_output must include feature_shapes and strides for decoding.")

        stride_values = torch.tensor(raw_output.strides, device=device, dtype=dtype)
        feature_maps = [
            torch.empty((1, 1, height, width), device=device, dtype=dtype)
            for height, width in raw_output.feature_shapes
        ]
        anchor_points, stride_tensor = make_anchors(feature_maps, stride_values, 0.5)
        anchors = anchor_points.transpose(0, 1).unsqueeze(0)
        strides = stride_tensor.transpose(0, 1)
        return anchors, strides

    def _decode_boxes(self, raw_output: YOLOERawOutput) -> torch.Tensor:
        head = self.model.get_detection_head()
        anchors, strides = self._generate_anchor_tensors(
            raw_output,
            device=raw_output.box_logits.device,
            dtype=raw_output.box_logits.dtype,
        )
        decoded = head.decode_bboxes(head.dfl(raw_output.box_logits), anchors) * strides
        return decoded

    def _postprocess_end2end(
        self,
        raw_output: YOLOERawOutput,
        decoded_boxes: torch.Tensor,
        *,
        max_det: int,
    ) -> list[torch.Tensor]:
        head = self.model.get_detection_head()
        scores = raw_output.class_logits.sigmoid().permute(0, 2, 1)
        boxes = decoded_boxes.permute(0, 2, 1)
        score_values, class_indices, gather_index = head.get_topk_index(scores, max_det)
        gathered_boxes = boxes.gather(dim=1, index=gather_index.repeat(1, 1, 4))
        outputs = [torch.cat([gathered_boxes, score_values, class_indices], dim=-1)]

        if raw_output.mask_coefficients is not None:
            mask_coefficients = raw_output.mask_coefficients.permute(0, 2, 1)
            gathered_masks = mask_coefficients.gather(
                dim=1,
                index=gather_index.repeat(1, 1, mask_coefficients.shape[-1]),
            )
            outputs = [torch.cat([gathered_boxes, score_values, class_indices, gathered_masks], dim=-1)]

        return [prediction for prediction in outputs[0]]

    def _postprocess_legacy(
        self,
        raw_output: YOLOERawOutput,
        decoded_boxes: torch.Tensor,
        *,
        max_det: int,
    ) -> list[torch.Tensor]:
        head = self.model.get_detection_head()
        prediction = torch.cat([decoded_boxes, raw_output.class_logits.sigmoid()], dim=1)
        if raw_output.mask_coefficients is not None:
            prediction = torch.cat([prediction, raw_output.mask_coefficients], dim=1)
        return nms.non_max_suppression(
            prediction=prediction,
            conf_thres=float(self.config.score_threshold),
            iou_thres=float(self.config.iou_threshold),
            agnostic=True,
            max_det=max_det,
            nc=int(head.nc),
            end2end=False,
        )

    def _filter_by_score(
        self,
        predictions: list[torch.Tensor],
        *,
        score_threshold: float,
    ) -> list[torch.Tensor]:
        filtered: list[torch.Tensor] = []
        for prediction in predictions:
            if prediction.numel() == 0:
                filtered.append(prediction)
                continue
            keep = prediction[:, 4] >= score_threshold
            filtered.append(prediction[keep])
        return filtered

    def _convert_predictions(
        self,
        predictions: list[torch.Tensor],
        device: torch.device,
        *,
        raw_output: YOLOERawOutput,
    ) -> RawYoloOutput:
        boxes_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor | None] = []

        for prediction in predictions:
            if prediction.numel() == 0:
                box_tensor, score_tensor, label_tensor, mask_tensor = self._empty_output(device)
                boxes_list.append(box_tensor)
                scores_list.append(score_tensor)
                labels_list.append(label_tensor)
                masks_list.append(mask_tensor)
                continue

            xyxy = prediction[:, :4].detach().to(device=device, dtype=torch.float32)
            conf = prediction[:, 4].detach().to(device=device, dtype=torch.float32)
            cls = prediction[:, 5].detach().to(device=device, dtype=torch.long)

            mask_tensor = None
            if raw_output.mask_coefficients is not None and prediction.shape[1] > 6:
                mask_tensor = prediction[:, 6:].detach().to(device=device, dtype=torch.float32)

            boxes_list.append(xyxy)
            scores_list.append(conf)
            labels_list.append(cls)
            masks_list.append(mask_tensor)

        return RawYoloOutput(
            boxes=boxes_list,
            scores=scores_list,
            labels=labels_list,
            masks=masks_list,
            raw_results=raw_output,
        )

    def _decode_raw_output(
        self,
        raw_output: YOLOERawOutput,
        *,
        max_det: int,
    ) -> list[torch.Tensor]:
        decoded_boxes = self._decode_boxes(raw_output)
        head = self.model.get_detection_head()
        if head.end2end:
            return self._postprocess_end2end(
                raw_output,
                decoded_boxes,
                max_det=max_det,
            )
        return self._postprocess_legacy(
            raw_output,
            decoded_boxes,
            max_det=max_det,
        )

    def _strip_mask_outputs(self, raw_output: YOLOERawOutput) -> YOLOERawOutput:
        if raw_output.mask_coefficients is None and raw_output.prototypes is None:
            return raw_output
        return YOLOERawOutput(
            box_logits=raw_output.box_logits,
            class_logits=raw_output.class_logits,
            mask_coefficients=None,
            prototypes=None,
            feature_shapes=raw_output.feature_shapes,
            strides=raw_output.strides,
            text_embeddings=raw_output.text_embeddings,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        text_embeddings: torch.Tensor,
        max_det: int | None = None,
        include_masks: bool = True,
        **_,
    ) -> RawYoloOutput:
        pixel_values = self._validate_pixel_values(pixel_values).contiguous()
        text_embeddings = self._validate_text_embeddings(text_embeddings).contiguous()
        requested_max_det = int(max_det or 300)
        if requested_max_det <= 0:
            requested_max_det = 300

        raw_output = self.model(
            pixel_values=pixel_values,
            text_embeddings=text_embeddings,
            include_masks=include_masks,
        )
        if not include_masks:
            raw_output = self._strip_mask_outputs(raw_output)
        filtered_predictions = self._filter_by_score(
            self._decode_raw_output(raw_output, max_det=requested_max_det),
            score_threshold=float(self.config.score_threshold),
        )
        return self._convert_predictions(
            filtered_predictions,
            pixel_values.device,
            raw_output=raw_output,
        )


class CudaGraphYoloRuntime(BaseYoloRuntime):
    def __init__(self, model: YOLOEModel) -> None:
        super().__init__(model)
        self._cuda_graph_cache: dict[tuple[object, ...], _CUDAGraphState] = {}

    def _capture_cuda_graph(
        self,
        pixel_values: torch.Tensor,
        text_embeddings: torch.Tensor,
        *,
        include_masks: bool,
    ) -> _CUDAGraphState:
        with torch.cuda.device(pixel_values.device):
            static_input = torch.empty_like(pixel_values, memory_format=torch.contiguous_format)
            static_input.copy_(pixel_values)
            static_text_embeddings = torch.empty_like(text_embeddings, memory_format=torch.contiguous_format)
            static_text_embeddings.copy_(text_embeddings)

            warmup_stream = torch.cuda.Stream(device=pixel_values.device)
            current_stream = torch.cuda.current_stream(device=pixel_values.device)
            warmup_stream.wait_stream(current_stream)

            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    self.model(
                        pixel_values=static_input,
                        text_embeddings=static_text_embeddings,
                        include_masks=include_masks,
                    )

            current_stream.wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self.model(
                    pixel_values=static_input,
                    text_embeddings=static_text_embeddings,
                    include_masks=include_masks,
                )

        return _CUDAGraphState(
            graph=graph,
            static_input=static_input,
            static_text_embeddings=static_text_embeddings,
            static_output=static_output,
        )

    def _run_raw_output(
        self,
        pixel_values: torch.Tensor,
        text_embeddings: torch.Tensor,
        *,
        include_masks: bool,
    ) -> YOLOERawOutput:
        if not pixel_values.is_cuda:
            return self.model(
                pixel_values=pixel_values,
                text_embeddings=text_embeddings,
                include_masks=include_masks,
            )

        graph_key = (
            pixel_values.device.index,
            pixel_values.dtype,
            tuple(pixel_values.shape),
            text_embeddings.dtype,
            tuple(text_embeddings.shape),
            bool(include_masks),
        )
        graph_state = self._cuda_graph_cache.get(graph_key)
        if graph_state is None:
            graph_state = self._capture_cuda_graph(
                pixel_values=pixel_values,
                text_embeddings=text_embeddings,
                include_masks=include_masks,
            )
            self._cuda_graph_cache[graph_key] = graph_state

        with torch.cuda.device(pixel_values.device):
            graph_state.static_input.copy_(pixel_values)
            graph_state.static_text_embeddings.copy_(text_embeddings)
            graph_state.graph.replay()
        return graph_state.static_output

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        text_embeddings: torch.Tensor,
        max_det: int | None = None,
        include_masks: bool = True,
        **_,
    ) -> RawYoloOutput:
        pixel_values = self._validate_pixel_values(pixel_values).contiguous()
        text_embeddings = self._validate_text_embeddings(text_embeddings).contiguous()
        requested_max_det = int(max_det or 300)
        if requested_max_det <= 0:
            requested_max_det = 300

        raw_output = self._run_raw_output(
            pixel_values=pixel_values,
            text_embeddings=text_embeddings,
            include_masks=include_masks,
        )
        if not include_masks:
            raw_output = self._strip_mask_outputs(raw_output)
        filtered_predictions = self._filter_by_score(
            self._decode_raw_output(raw_output, max_det=requested_max_det),
            score_threshold=float(self.config.score_threshold),
        )
        return self._convert_predictions(
            filtered_predictions,
            pixel_values.device,
            raw_output=raw_output,
        )


__all__ = ["BaseYoloRuntime", "CudaGraphYoloRuntime", "RawYoloOutput"]
