"""Optional OpenTelemetry tracing (enabled via OTEL_ENABLED=true)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from src.core.config import get_settings

_tracer: Any = None
_initialized = False


def setup_tracing(app: Any | None = None) -> bool:
    """Initialize OTLP tracing when enabled. Returns True if active."""
    global _tracer, _initialized
    settings = get_settings()
    if not settings.otel_enabled:
        return False
    if _initialized:
        return _tracer is not None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        _initialized = True
        return False

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)

    if settings.otel_exporter_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(settings.otel_service_name)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            pass
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
        except ImportError:
            pass

    _initialized = True
    return _tracer is not None


@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[None]:
    """Create a span when tracing is enabled; no-op otherwise."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, str(value))
        yield
