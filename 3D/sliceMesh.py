import vedo
from vedo import load

def sliceMesh(widget, event):
    global current_slice, alphaValue
    z = widget.value

    if current_slice is not None:
        plotter.remove(current_slice)

    if mesh in plotter.get_meshes():
        plotter.remove(mesh)

    new_slice = mesh.clone().cut_with_plane(origin=(0,0,z), normal="z")
    new_slice.alpha(alphaValue)

    plotter.add(new_slice)
    current_slice = new_slice

    plotter.render()

def alphaMesh(widget, event):
    global current_slice, alphaValue
    alphaValue = widget.value

    if mesh in plotter.get_meshes():
        mesh.alpha(alphaValue)
        plotter.render()
    
    if current_slice is not None:
        current_slice.alpha(alphaValue)
        plotter.render()

def savePlotter(widget, event):
    global sliceSlider, alphaSlider, saveButton
    sliceSlider.GetRepresentation().VisibilityOff()
    alphaSlider.GetRepresentation().VisibilityOff()
    saveButton.VisibilityOff()
    plotter.render()

    plotter.screenshot("Images/sliced_mesh.png")
    print("Screenshot saved to Images/sliced_mesh.png")

    sliceSlider.GetRepresentation().VisibilityOn()
    alphaSlider.GetRepresentation().VisibilityOn()
    saveButton.VisibilityOn()
    plotter.render()

if __name__ == '__main__':
    alphaValue = 1.0
    current_slice = None

    mesh = load("mesh/mesh_1.stl")
    mesh.alpha(alphaValue) 

    originalMesh = mesh.clone()

    plotter = vedo.Plotter()
    plotter.roll(90)
    plotter.azimuth(90)
    plotter.add(mesh)
    sliceSlider = plotter.add_slider(sliceMesh, 0, 512, value=0, title="Slice Mesh", pos="top-right")
    alphaSlider = plotter.add_slider(alphaMesh, 0, 1.0, value=alphaValue, title="Alpha", pos="top-left")
    saveButton = plotter.add_button(savePlotter, states=["Save"], pos=(0.9, 0.8), size=20)
    plotter.show()
