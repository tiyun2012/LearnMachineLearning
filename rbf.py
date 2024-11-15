import numpy as np

import maya.cmds as cmds

def get_selected_objects_positions():
    """
    Returns a list of positions for the currently selected objects in Maya.
    
    Returns:
    list of tuples: A list where each element is a tuple (x, y, z) representing the position of an object.
    """
    # Get the list of selected objects
    selected_objects = cmds.ls(selection=True)
    
    # List to store positions
    positions = []
    
    # Loop through each selected object and get its world position
    for obj in selected_objects:
        pos = cmds.xform(obj, query=True, worldSpace=True, translation=True)
        positions.append(tuple(pos))
    
    return positions

# Example usage:
positions = get_selected_objects_positions()
print(positions)


def distance_matrix_2d(vectors):
    """
    Returns an n x n matrix where each element (i, j) is the Euclidean distance between (xi, yi) and (xj, yj).

    Parameters:
    vectors (list of tuples): List of 3D vectors [(xi, yi, zi)].

    Returns:
    numpy.ndarray: n x n distance matrix.
    """
    n = len(vectors)
    distance_matrix = np.zeros((n, n))
    
    # Loop through each pair of vectors to calculate 2D distance between (xi, yi) and (xj, yj)
    for i in range(n):
        for j in range(n):
            # Calculate 2D distance between (xi, yi) and (xj, yj)
            distance = np.sqrt((vectors[i][0] - vectors[j][0]) ** 2 + (vectors[i][2] - vectors[j][2]) ** 2)
            distance_matrix[i][j] = distance
    
    return distance_matrix

distanceMatrix = distance_matrix_2d(positions)
print(distanceMatrix)

def inverse_matrix(matrix):
    try:
        # Use NumPy's `linalg.inv` function to calculate the inverse
        inv_matrix = np.linalg.inv(matrix)
        return inv_matrix
    except np.linalg.LinAlgError:
        # This exception is raised if the matrix is singular and cannot be inverted
        return "Matrix is singular and cannot be inverted."

distanceMatrix_inverse=inverse_matrix(distanceMatrix)
def find_min_e_for_invertibility(A, tolerance=1e-10, max_iter=1000):
    """
    Finds the minimum value of e such that A + eI is invertible.
    
    Parameters:
    A (numpy.ndarray): Input square matrix.
    tolerance (float): Precision threshold for determinant check.
    max_iter (int): Maximum number of iterations to find e.
    
    Returns:
    e (float): Minimum value such that A + eI is invertible.
    A_inv (numpy.ndarray): The inverse of A + eI.
    """
    n = A.shape[0]
    I = np.eye(n)  # Identity matrix
    e = 0.0  # Start with no regularization

    for i in range(max_iter):
        try:
            # Try to calculate the inverse of A + eI
            A_inv = np.linalg.inv(A + e * I)
            return e, A_inv
        except np.linalg.LinAlgError:
            # Matrix is singular, increase e and try again
            e += tolerance  # Increment e by a small amount
            
    return None, None  # If no solution is found after max_iter

distanceMatrix_inverse=find_min_e_for_invertibility(distanceMatrix)[1]




def extract_z_component(vectors):
    """
    Returns a numpy array of the z-components from a list of 3D vectors.
    
    Parameters:
    vectors (list of tuples): List of 3D vectors [(xi, yi, zi)].
    
    Returns:
    numpy.ndarray: Array of z-components [zi].
    """
    # Extract the z-component from each vector
    z_components = np.array([vector[2] for vector in vectors])
    return z_components

# Example usage:
z_result = extract_z_component(positions)


def transpose_and_multiply(A, B):
    """
    Returns A^T * B if A is not None.
    
    Parameters:
    A (numpy.ndarray or None): Matrix A.
    B (numpy.ndarray): Matrix B.
    
    Returns:
    numpy.ndarray or str: Result of A^T * B, or a message if A is None.
    """
    if A is not None:
        try:
            result = np.dot(A, B.T)
            return result
        except ValueError as e:
            return f"Error in matrix multiplication: {str(e)}"
    else:
        return "A is None, cannot perform operation."

# Example usage:

weights = transpose_and_multiply(distanceMatrix_inverse, z_result)

import math

def compute_f(w, x, y, points,test='pCube1'):
    """
    Computes the sum of wi * sqrt((x - xi)^2 + (y - yi)^2) for i=1..n.
    
    Parameters:
    w (list of floats): List of weights [w1, w2, ..., wn].
    x (float): The x coordinate of the point.
    y (float): The y coordinate of the point.
    points (list of tuples): List of points [(x1, y1), (x2, y2), ..., (xn, yn)].
    
    Returns:
    float: The computed sum f.
    """
    # Make sure the length of weights matches the number of points
    assert len(w) == len(points), "The number of weights must match the number of points."
    
    point2d=[]
    for i in points:
        point2d.append((i[0],i[1])) 
    # Calculate the sum
    z = sum(wi * math.sqrt((x - xi)**2 + (y - yi)**2) for wi, (xi, yi) in zip(w, point2d))
    if test:
    	cmds.setAttr(test+'.translateX',x)
    	cmds.setAttr(test+'.translateY',y)
    	cmds.setAttr(test+'.translateZ',float(z))
    return z



result = compute_f(weights, 1.962,1.714, positions)
def computeMesh(w,points,test='pPlaneShape1'):

    # Make sure the length of weights matches the number of points
    assert len(w) == len(points), "The number of weights must match the number of points."
try:
    selection_list = om.MSelectionList()
    selection_list.add(test)  
    dag_path = selection_list.getDagPath(0)  
except Exception as e:
    om.MGlobal.displayError(f"Error: {str(e)}")
    return None	
if dag_path.node().hasFn(om.MFn.kMesh):
    # Create the MFnMesh function set from the MDagPath
    mesh_fn = om.MFnMesh(dag_path)
else:
    om.MGlobal.displayError(f"{test} is not a mesh.")
    return None  	        
# Calculate the sum
points = mesh_fn.getPoints(om.MSpace.kWorld)
for i in range(len(points)):
    x=points[i].y
    z=points[i].z
    y = sum(wi * math.sqrt((x - xi)**2 + (z - zi)**2) for wi, (xi, zi) in zip(w, point2d))    	
    points[i].y = float(y) 
mesh_fn.setPoints(points, om.MSpace.kWorld)

computeMesh(weights,positions)    
    