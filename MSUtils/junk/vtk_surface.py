import numpy as np
import vtk
from vtk import vtkWindowedSincPolyDataFilter
from vtk.util import numpy_support


def get_mesh_normals(polydata):
    """
    Get normals of a VTK polydata mesh that are pointing outwards.

    Parameters:
    - polydata (vtk.vtkPolyData): Input polydata mesh.

    Returns:
    - vtk.vtkPolyData: Mesh with normals.
    """
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.AutoOrientNormalsOn()  # Automatically re-orient normals
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def array_to_vtk_image(data_array):
    """
    Convert a numpy array to a VTK image format.

    Parameters:
    - data_array (numpy.ndarray): Input 3D numpy array containing labeled data.

    Returns:
    - vtk.vtkImageData: Converted VTK image data.
    """
    # Convert the numpy array to a VTK data format
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=data_array.ravel(), deep=True, array_type=vtk.VTK_INT
    )

    # Create an empty VTK image data object
    image = vtk.vtkImageData()

    # Set the dimensions of the VTK image to match the numpy array
    image.SetDimensions(data_array.shape)

    # Attach the converted data to the VTK image
    image.GetPointData().SetScalars(vtk_data)

    return image


def extract_surfaces_from_array(data_array):
    """
    Extract surface meshes for all unique labels present in the data array using VTK's Marching Cubes algorithm.

    Parameters:
    - data_array (numpy.ndarray): Input 3D numpy array containing labeled data.

    Returns:
    - vtk.vtkPolyData: Combined surface mesh of all unique regions in the input data.
    """
    # Convert the numpy array to VTK image data
    data = array_to_vtk_image(data_array)

    # Identify all unique labels in the data
    unique_labels = np.unique(data_array)

    surfaces = []
    for label in unique_labels:
        # Initialize the discrete marching cubes algorithm for surface extraction
        dmc = vtk.vtkDiscreteMarchingCubes()

        # Provide the VTK image data as input
        dmc.SetInputData(data)

        # Generate values for the current label
        dmc.GenerateValues(1, label, label)

        # Execute the marching cubes algorithm
        dmc.Update()

        # Correct the normals
        polydata = get_mesh_normals(dmc.GetOutput())

        # Collect the resulting surface mesh
        surfaces.append(polydata)

    # Initialize a filter to append multiple polydata objects into one
    append_filter = vtk.vtkAppendPolyData()
    for surface in surfaces:
        append_filter.AddInputData(surface)
    append_filter.Update()

    return append_filter.GetOutput()


def save_polydata_to_vtk(polydata, filename):
    """
    Save a VTK polydata object to a .vtk file format.

    Parameters:
    - polydata (vtk.vtkPolyData): Input polydata to be saved.
    - filename (str): Path to the output .vtk file.
    """
    # Initialize a VTK polydata writer
    writer = vtk.vtkPolyDataWriter()

    # Set the input polydata
    writer.SetInputData(polydata)

    # Specify the output file name
    writer.SetFileName(filename)

    # Write the polydata to file
    writer.Write()


def smooth_mesh(polydata, iterations=20, pass_band=0.1, boundary_smoothing=True):
    """
    Smooth the input mesh using the Windowed Sinc algorithm in VTK.

    Parameters:
    - polydata (vtk.vtkPolyData): Input polydata mesh.
    - iterations (int): Number of smoothing iterations. Default is 20.
    - pass_band (float): Passband value for the smoothing filter. Default is 0.1.
    - boundary_smoothing (bool): Whether to smooth the boundaries. Default is True.

    Returns:
    - vtk.vtkPolyData: Smoothed mesh.
    """
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOn() if boundary_smoothing else smoother.BoundarySmoothingOff()
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    return smoother.GetOutput()
