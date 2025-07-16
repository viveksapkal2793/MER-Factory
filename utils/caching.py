# -*- coding: utf-8 -*-
# Author: Yuxiang Lin (with caching additions)

import hashlib
from pathlib import Path
from functools import wraps
import diskcache
from rich.console import Console
import asyncio

console = Console(stderr=True)


def get_file_hash(file_path: Path) -> str:
    """Computes the SHA256 hash of a file to use in a cache key."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K for efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


def create_cache_key(model_name: str, func_name: str, args, kwargs) -> tuple:
    """
    Creates a robust cache key from the function's arguments.
    It handles file paths by hashing their content.
    """
    key_parts = [model_name, func_name]

    # Process positional arguments
    for arg in args:
        if isinstance(arg, Path):
            key_parts.append(f"path_hash:{get_file_hash(arg)}")
        elif isinstance(arg, (str, int, float, bool)):
            key_parts.append(arg)
        else:
            # For other types, use their string representation as a fallback.
            key_parts.append(str(arg))

    # Process keyword arguments, sorted by key for consistency
    for key, value in sorted(kwargs.items()):
        key_parts.append(key)
        if isinstance(value, Path):
            key_parts.append(f"path_hash:{get_file_hash(value)}")
        elif isinstance(value, (str, int, float, bool)):
            key_parts.append(value)
        else:
            key_parts.append(str(value))

    return tuple(key_parts)


def cache_llm_call(cache: diskcache.Cache):
    """
    A decorator factory that caches the result of an LLM model method.
    It can wrap both synchronous and asynchronous functions.
    """

    def decorator(func):
        # Common logic to get the cache key and model identifier
        def get_cache_key_and_id(*args, **kwargs):
            # The decorator wraps a bound method, so the instance is func.__self__
            model_instance = func.__self__
            model_identifier = (
                getattr(model_instance, "model_name", None)
                or getattr(model_instance, "model_id", None)
                or model_instance.__class__.__name__
            )
            key = create_cache_key(model_identifier, func.__name__, args, kwargs)
            return key, model_identifier

        # Check if the function to be decorated is async and return the appropriate wrapper
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                key, model_id = get_cache_key_and_id(*args, **kwargs)
                if key in cache:
                    console.log(
                        f"LLM Cache HIT for '[bold yellow]{func.__name__}[/bold yellow]' on model '[bold cyan]{model_id}[/bold cyan]'"
                    )
                    return cache.get(key)

                console.log(
                    f"LLM Cache MISS for '[bold yellow]{func.__name__}[/bold yellow]' on model '[bold cyan]{model_id}[/bold cyan]'. Calling API."
                )
                result = await func(*args, **kwargs)
                cache.set(key, result)
                return result

            return async_wrapper
        else:
            # It's a synchronous function
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                key, model_id = get_cache_key_and_id(*args, **kwargs)
                if key in cache:
                    console.log(
                        f"LLM Cache HIT for '[bold yellow]{func.__name__}[/bold yellow]' on model '[bold cyan]{model_id}[/bold cyan]'"
                    )
                    return cache.get(key)

                console.log(
                    f"LLM Cache MISS for '[bold yellow]{func.__name__}[/bold yellow]' on model '[bold cyan]{model_id}[/bold cyan]'. Calling API."
                )
                result = func(*args, **kwargs)
                cache.set(key, result)
                return result

            return sync_wrapper

    return decorator
