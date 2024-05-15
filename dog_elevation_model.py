import numpy as np
import pyvista as pv
from pyvista.plotting.utilities import start_xvfb
from PIL import Image

import streamlit as st
from stpyvista import stpyvista
from stpyvista.utils import is_the_app_embedded


# Generate plotter
@st.cache_resource
def dog_texture(dummy: str = "dog"):
    PATH_TO_JPG = "./assets/img/gloria_pickle.jpg"
    tex = pv.read_texture(PATH_TO_JPG)

    with Image.open(PATH_TO_JPG) as im:
        gray_scale = im.convert(mode="L").resize([x // 2 for x in im.size])
        width, height = gray_scale.size

    # Create mesh grid
    x = np.arange(width)
    y = np.arange(height, 0, -1)
    xx, yy = np.meshgrid(x, y)
    z = -0.25 * (np.array(gray_scale))

    # Generate surface
    surface = (
        pv.StructuredGrid(xx, yy, z)
        .triangulate()
        .extract_surface()
        .smooth()
        .texture_map_to_plane(use_bounds=True, inplace=True)
    )

    # Lower elevations -> transparent
    zp = surface.points[:, 2]
    opacity = np.interp(zp, [zp.min(), 0.90 * zp.max()], [0, 1])

    # Assemble plotter
    plotter = pv.Plotter()
    plotter.window_size = [400, 350]
    plotter.background_color = "#efe4cf"

    plotter.add_mesh(
        surface,
        texture=tex,
        show_scalar_bar=False,
        opacity=opacity,
        name="dog",
    )

    # Zooming and camera configs
    plotter.camera_position = "xy"
    plotter.camera.elevation = -10
    pos = plotter.camera.position
    fcp = plotter.camera.focal_point
    plotter.camera.position = [0.65 * p + 0.35 * f for p, f in zip(pos, fcp)]

    # Last touches
    plotter.add_text(
        "🐾",
        position="upper_left",
        color="black",
        font_size=18,
        shadow=True,
    )

    return plotter


# Initial configuration
@st.cache_resource
def initial_config(dummy:str = "init"):
    start_xvfb()
    return

@st.cache_resource
def get_css(dummy:str = "css"):
    css_list = []
    for style_sheet in ["./assets/style.css", "./assets/style_embed.css"]:
        with open(style_sheet) as f:
            css_list.append(f.read())

    return css_list


def main():
    
    st.set_page_config(
        page_title="stpyvista: Dog Canyon",
        page_icon="🐕",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    initial_config()

    st.session_state.is_app_embedded = st.session_state.get(
        "is_app_embedded", is_the_app_embedded()
    )
    
    # Add some styling with CSS selectors
    for css in get_css():
        st.html(f"<style>{css}</style>")
    
    # Load pyvista Plotter
    dog = dog_texture()

    if not st.session_state.is_app_embedded:
        
        st.header("🐕   Dog Elevation Model (DEM)", divider="rainbow")
        "&nbsp;"

        cols = st.columns([1, 3])
        
        with cols[0]:
            "Generate a digital elevation model from an image's brightness."
            st.image("./assets/img/gloria_pickle.jpg")
            st.caption(
                "Gloria from [The Coding Train](https://thecodingtrain.com/challenges/181-image-stippling)"
            )
        
        with cols[1]:
            container_3d = st.container()

        st.html(
            """
            <style>
                p{
                    text-align: center; 
                    line-height: 1em;
                }
                div[data-testid="stImage"] img{
                    border: 7px ridge rgba(211, 220, 50, .6);
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: max(25%, 12vw);
                }
            </style>
            """
        )
        
    else:
        container_3d = st.container()

    with container_3d:
        stpyvista(
            dog,
            panel_kwargs=dict(
                orientation_widget=True, interactive_orientation_widget=True
            ),
            bokeh_resources="CDN",
            use_container_width=True,
            key="dem-dog"
        )

    if st.session_state.is_app_embedded:
        st.html(
            """
            <style>
                body{
                    background-color: rgba(0,0,0,0);
                }
                div[data-testid="stAppViewBlockContainer"]{
                    padding: 0; 
                }
                iframe[title="stpyvista.rendered"]{
                    display: block; 
                    width: 95vw; 
                    border: none;
                }

                div[data-testid="stVerticalBlockBorderWrapper"]{
                    transform: translateY(-12px);
                }
            </style>
            """
        )

if __name__ == "__main__":
    main()
