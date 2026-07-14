from __future__ import annotations

import asyncio
from typing import Any

from src.core.config import Settings, get_settings
from src.utils.text import sanitize_assistant_reply


class LocalQwenService:
    """Singleton local Qwen model with inference queue to prevent GPU OOM."""

    _instance: LocalQwenService | None = None

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None
        self._tokenizer = None
        self._queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None
        self._loaded = False

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> LocalQwenService:
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from modelscope import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            raise RuntimeError(
                "GPU dependencies not installed. Install with: pip install zhineng-kefu[gpu]"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.settings.qwen_model_path, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.settings.qwen_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.settings.lora_path_exists:
            model = PeftModel.from_pretrained(base_model, self.settings.lora_path)
            self._model = model.merge_and_unload()
        else:
            self._model = base_model

        self._model.eval()
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._queue = None
        self._worker_task = None
        LocalQwenService._instance = None

    async def shutdown(self) -> None:
        """Stop background worker and release model memory."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self.unload()
        try:
            import gc

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except ImportError:
            pass

    async def start_worker(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._inference_worker())

    async def _inference_worker(self) -> None:
        while True:
            future, fn, args, kwargs = await self._queue.get()
            try:
                result = await asyncio.to_thread(fn, *args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self._queue.task_done()

    async def _enqueue(self, fn, *args, **kwargs) -> Any:
        if self._queue is None:
            await self.start_worker()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._queue.put((future, fn, args, kwargs))
        return await future

    def _generate_sync(self, prompt: str) -> str:
        import torch

        if not self._loaded:
            self.load()

        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
            )
        output_ids = generated_ids[0][len(inputs.input_ids[0]) :]
        raw = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return sanitize_assistant_reply(raw)

    def _generate_from_messages_sync(self, messages: list[dict]) -> str:
        if not self._loaded:
            self.load()
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self._generate_sync(prompt)

    async def generate(self, prompt: str) -> str:
        return await self._enqueue(self._generate_sync, prompt)

    def _generate_from_messages_stream_sync(self, messages: list[dict]):
        """Sync generator yielding decoded tokens from local Qwen."""
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer

        if not self._loaded:
            self.load()

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def _run_generate():
            with torch.no_grad():
                self._model.generate(
                    **inputs,
                    max_new_tokens=self.settings.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.85,
                    streamer=streamer,
                )

        thread = Thread(target=_run_generate, daemon=True)
        thread.start()
        for token in streamer:
            if token:
                yield token
        thread.join(timeout=120)

    async def generate_from_messages_stream(self, messages: list[dict]):
        """Async token stream for RAG answers (true streaming from local Qwen)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

        def _producer():
            try:
                for token in self._generate_from_messages_stream_sync(messages):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        await self.start_worker()
        Thread = __import__("threading").Thread
        Thread(target=_producer, daemon=True).start()

        while True:
            kind, value = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                raise RuntimeError(value or "Qwen stream failed")
            yield value

    async def generate_from_messages(self, messages: list[dict]) -> str:
        return await self._enqueue(self._generate_from_messages_sync, messages)
