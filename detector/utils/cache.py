# cache.py
import os
import time
import pickle
import threading
from typing import Any, Callable, Optional
from filelock import FileLock

class Cache:
    """
    Thread-safe & process-safe persistent cache:
    - In-memory dict guarded by threading.Lock
    - Cross-process file write protected by FileLock
    - Atomic dump via temp file + os.replace
    """
    def __init__(
        self,
        cache_file: str,
        key_normalizer: Optional[Callable[[str], str]] = None,
        retry_on_load_error: bool = True,
    ):
        self.cache_file = cache_file
        self.lockfile = cache_file + ".lock"
        self.dir = os.path.dirname(cache_file) or "."
        os.makedirs(self.dir, exist_ok=True)

        self._lock = threading.Lock()
        self._add_n = 0
        self._norm = key_normalizer or (lambda s: s.strip())

        self._cache = self._load(retry_on_error=retry_on_load_error)

    # ---------- Public API ----------
    def get(self, key: str) -> Optional[Any]:
        k = self._norm(key)
        with self._lock:
            return self._cache.get(k, None)

    # 原语义别名：兼容旧代码 check_prompt(prompt) -> Optional[Any]
    def check_prompt(self, prompt: str) -> Optional[Any]:
        return self.get(prompt)

    def set(self, key: str, value: Any) -> None:
        k = self._norm(key)
        with self._lock:
            self._cache[k] = value
            self._add_n += 1

    # 原语义别名：兼容旧代码 save_prompt(prompt, response)
    def save_prompt(self, prompt: str, response: Any) -> None:
        self.set(prompt, response)

    def get_pending_count(self) -> int:
        with self._lock:
            return self._add_n

    def flush(self) -> None:
        """Force write all current data to disk (merge with on-disk)."""
        self._flush_impl(force=True)

    def maybe_flush(self, threshold: int = 8) -> bool:
        """
        If pending updates >= threshold, flush once.
        Returns True if a flush happened, else False.
        """
        return self._flush_impl(force=False, threshold=threshold)

    def close(self) -> None:
        """Flush and release (idempotent)."""
        try:
            self.flush()
        except Exception:
            # 不要因为关库失败而让主流程崩
            pass

    # 支持 with 使用，确保退出时落盘
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------- Internal I/O ----------
    def _flush_impl(self, force: bool, threshold: int = 8) -> bool:
        # 1) 取快照 + 复位计数（原子）
        with self._lock:
            if not force and self._add_n < threshold:
                return False
            snapshot = dict(self._cache)
            self._add_n = 0

        # 2) 进程级锁 + 合并写（锁外做 IO，避免长时间占用线程锁）
        lock = FileLock(self.lockfile)
        with lock:
            on_disk = {}
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "rb") as f:
                        on_disk = pickle.load(f)
                except Exception:
                    # 读盘失败，保守起见从空开始（也可选择 raise）
                    on_disk = {}

            # 合并：内存为主（last-writer-wins）
            on_disk.update(snapshot)

            # 原子写：写到 tmp 再替换
            tmp_path = self.cache_file + ".tmp"
            with open(tmp_path, "wb") as tmpf:
                pickle.dump(on_disk, tmpf, protocol=pickle.HIGHEST_PROTOCOL)
                tmpf.flush()
                os.fsync(tmpf.fileno())
            os.replace(tmp_path, self.cache_file)
        return True

    def _load(self, retry_on_error: bool = True, retry_sleep: float = 5.0):
        if not os.path.exists(self.cache_file):
            return {}
        while True:
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except EOFError:
                # 可能是之前非原子写导致的残缺；我们现在采用原子写，基本不会遇到
                if not retry_on_error:
                    raise
                # 自动修复为空
                with open(self.cache_file, "wb") as f:
                    pickle.dump({}, f, protocol=pickle.HIGHEST_PROTOCOL)
                return {}
            except Exception:
                if not retry_on_error:
                    raise
                # 等待一会再试（比如另一进程正在写）
                time.sleep(retry_sleep)
