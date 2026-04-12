from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from pathlib import Path

import grpc
import typer

from artemis_cve.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc
from artemis_cve.servicers import WebRtcDetectorServicer
from artemis_cve.utils.parse_name import parse_class_names

app = typer.Typer(name="artemis-cve")


@app.command()
def serve(
    model_path: str = typer.Argument(
        ...,
        envvar="MODEL_PATH",
        help="Local YOLOE model directory.",
    ),
    textencoder_model_path: str = typer.Argument(
        ...,
        envvar="TEXT_ENCODER_MODEL_PATH",
        help="Local text encoder model directory.",
    ),
    class_names_file: str | None = typer.Option(
        None,
        "--class-names-file",
        envvar="CLASS_NAMES_FILE",
        help="Path to a text file with one open-vocabulary class name per line.",
    ),
    device: str = typer.Option(
        "cpu",
        envvar="DEVICE",
        help="Inference device, for example cpu or cuda:0.",
    ),
    dtype: str = typer.Option(
        "fp32",
        envvar="DTYPE",
        help="Inference dtype: fp32, bf16, or fp16.",
    ),
    use_cuda_graph: bool = typer.Option(
        False,
        envvar="USE_CUDA_GRAPH",
        help="Enable CUDA Graph inference on CUDA devices.",
    ),
    host: str = typer.Option("0.0.0.0", envvar="GRPC_HOST", help="Listen address."),
    port: int = typer.Option(50051, envvar="GRPC_PORT", help="Listen port."),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("aioice.ice").setLevel(logging.WARNING)

    resolved_class_names = parse_class_names(class_names_file, Path(model_path))

    async def _run() -> None:
        server = grpc.aio.server()
        pb2_grpc.add_WebRtcDetectorEngineServicer_to_server(
            WebRtcDetectorServicer(
                model_dir=model_path,
                textencoder_model_dir=textencoder_model_path,
                class_names=resolved_class_names,
                device=device,
                dtype=dtype,
                use_cuda_graph=use_cuda_graph,
            ),
            server,
        )
        listen_addr = f"{host}:{port}"
        server.add_insecure_port(listen_addr)
        await server.start()
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(signum, stop_event.set)
        typer.echo(f"gRPC server listening on {listen_addr}")
        try:
            await stop_event.wait()
        finally:
            await server.stop(grace=1)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
