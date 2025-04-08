"""
Utilities for displaying Gradio interfaces within Jupyter or IPython environments.

This module provides helper functions to embed Gradio UIs using iframes.
"""

from IPython.display import HTML, display


def display_iframe(proxy_url: str, height: int = 700, width: str = "100%") -> None:
    """
    Display an iframe embedding the specified URL with configurable height and width.

    Args:
        proxy_url (str): The URL to be embedded in the iframe.
        height (int, optional): The height of the iframe in pixels. Defaults to 700.
        width (str, optional): The width of the iframe. Defaults to "100%".

    Returns:
        None
    """
    # Create the HTML iframe element with appropriate permissions and attributes
    artifact = HTML(
        f'<div><iframe src="{proxy_url}" width="{width}" height="{height}" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>'
    )

    # Display the iframe in the current IPython/Jupyter context
    display(artifact)
