#!/usr/bin/env python3
"""
Agentic Production House — CLI entry point.

Usage:
    python main.py --prompt "Create a relaxing travel vlog" --media-dir ./input
    python main.py --prompt "Product promo" --media ./clip1.mp4 ./img1.jpg
    python main.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Production House — AI-powered video production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py --prompt "Create a relaxing travel vlog" --media-dir ./input
  python main.py --prompt "Product promo video" --media clip1.mp4 clip2.mp4 product.jpg
  python main.py --prompt "Instagram reel" --media-dir ./clips --output reel.mp4
  python main.py --prompt "Wedding highlights" --media-dir ./wedding --music-dir ./music
        """,
    )

    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Natural-language description of the video to create",
    )
    parser.add_argument(
        "--media-dir", "-d",
        default=None,
        help="Directory containing media files (default: ./input)",
    )
    parser.add_argument(
        "--media", "-m",
        nargs="+",
        default=None,
        help="Explicit list of media file paths",
    )
    parser.add_argument(
        "--output", "-o",
        default="final_video.mp4",
        help="Output filename (default: final_video.mp4)",
    )
    parser.add_argument(
        "--music-dir",
        default=None,
        help="Directory with background music files",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio enhancement step",
    )
    parser.add_argument(
        "--generate-music",
        action="store_true",
        help="Synthesize instrumental BGM with MusicGen (needs: pip install -r requirements-musicgen.txt, GPU recommended)",
    )
    parser.add_argument(
        "--platform",
        choices=[
            "tiktok",
            "youtube_long",
            "youtube_shorts",
            "youtube_highlights",
            "vimeo_cinematic",
            "instagram_reels",
            "instagram_stories",
        ],
        default="youtube_long",
        help="Target platform profile (default: youtube_long)",
    )

    args = parser.parse_args()

    # Resolve media source
    media_dir = args.media_dir
    media_files = args.media

    if media_files is None and media_dir is None:
        default_input = Path("input")
        if default_input.exists() and any(default_input.iterdir()):
            media_dir = str(default_input)
        else:
            console.print(Panel(
                "[bold red]No media provided![/]\n\n"
                "Use [cyan]--media-dir[/] to point to a folder of clips/images,\n"
                "or [cyan]--media[/] to list specific files.\n\n"
                "Example:\n"
                "  python main.py --prompt \"travel vlog\" --media-dir ./input",
                title="Error",
            ))
            sys.exit(1)

    # Validate files exist
    if media_files:
        for f in media_files:
            if not Path(f).exists():
                console.print(f"[red]File not found:[/] {f}")
                sys.exit(1)
    elif media_dir:
        if not Path(media_dir).exists():
            console.print(f"[red]Directory not found:[/] {media_dir}")
            sys.exit(1)

    console.print(Panel(
        "[bold cyan]AGENTIC PRODUCTION HOUSE[/]\n"
        "AI-powered multi-agent video production",
        subtitle="v0.1.0",
    ))

    from src.orchestrator import ProductionOrchestrator

    orchestrator = ProductionOrchestrator(config_path=args.config)

    try:
        report = orchestrator.produce(
            prompt=args.prompt,
            media_dir=media_dir,
            media_files=media_files,
            output_filename=args.output,
            platform=args.platform,
            music_dir=args.music_dir,
            skip_audio_enhance=args.no_audio,
            generate_music=args.generate_music,
        )

        console.print()
        console.print(Panel(
            f"[bold green]✓ Video produced successfully![/]\n\n"
            f"  Output: [bold]{report.output_path}[/]\n"
            f"  Duration: {report.duration:.1f}s\n"
            f"  Resolution: {report.resolution}\n"
            f"  Processing time: {report.processing_time_seconds:.1f}s",
            title="Complete",
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        console.print_exception()
        sys.exit(1)
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
