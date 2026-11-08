"""Notebook-only helpers.

[GameEnv] IPython / ipywidgets are imported lazily inside the functions so
that importing GameEnv does not require the Jupyter stack.
"""


def show_image_in_ipython(data, *args, **kwargs):
    from IPython.display import Image, display

    display(Image(data, *args, **kwargs))


def ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    from IPython.display import Image, display
    from ipywidgets import IntSlider, interactive

    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))

    return interactive(
        display_f,
        **{slider_label: IntSlider(min=0, max=len(image_pathes_list) - 1, step=1)},
    )


def show_ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    from IPython.display import Image, display
    from ipywidgets import IntSlider, interactive

    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))

    display(
        interactive(
            display_f,
            **{slider_label: IntSlider(min=0, max=len(image_pathes_list) - 1, step=1)},
        )
    )
