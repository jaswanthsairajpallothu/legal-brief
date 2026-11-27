"""Judora Gradio application entry point."""
from ui.gradio_app import create_app, launch

demo = create_app()


if __name__ == "__main__":
    launch(demo)
