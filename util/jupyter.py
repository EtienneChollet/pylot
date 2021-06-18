from IPython.display import HTML, display

from .more_functools import static_vars

@static_vars(initialized=False)
def notebook_put_into_clipboard(text):

    if not notebook_put_into_clipboard.initialized:

        init = HTML("""
        <script>
        function updateClipboard(newClip) {
          navigator.clipboard.writeText(newClip).then(function() {
            /* clipboard successfully set */
          }, function() {
            /* clipboard write failed */
          });
        }
        </script>""")
        display(init)

    display(HTML(f'<script> updateClipboard("{text}")  </script>'))


def disable_widget_scroll():
    style = """
    <style>
       .jupyter-widgets-output-area .output_scroll {
            height: unset !important;
            border-radius: unset !important;
            -webkit-box-shadow: unset !important;
            box-shadow: unset !important;
        }
        .jupyter-widgets-output-area  {
            height: auto !important;
        }
    </style>
    """
    display(HTML(style))


def widget_dark_mode():
    style = """
    <style>
        .widget-label-basic {
            color: #ddd;
        }
    </style>
    """
    display(HTML(style))


def jupyter_width(fraction):
    percent = int(fraction * 100)
    style = f"""
    <style>
        .container {{
            width: {percent:d}% !important;
        }}
    </style>
    """
    display(HTML(style))
