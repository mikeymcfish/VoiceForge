from __future__ import annotations

from .app import create_app


def main() -> None:
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
