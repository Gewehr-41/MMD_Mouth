"""Lyrics -> viseme timeline conversion.

The recognition backends (Vosk, worker process) belonged to the Blender add-on
and were removed together with it.  What remains is the part the standalone
tool uses: parsing LRC/QRC/plain text, tagging languages and turning lines into
viseme events.
"""

from .lyrics import LyricsError, run_lyrics_pipeline

__all__ = ["LyricsError", "run_lyrics_pipeline"]
