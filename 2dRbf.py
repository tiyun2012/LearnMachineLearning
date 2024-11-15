import numpy as np

# Helper function to create a 2D rotation matrix
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

# Helper function to create a 2D scaling matrix
def scaling_matrix(sx, sy):
    return np.array([[sx, 0],
                     [0, sy]])

# Helper function to create a 2D translation matrix (as part of a homogeneous matrix)
def translation_matrix(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

# Step 1: Generate transformation matrices (flattened)
rotation = rotation_matrix(np.pi / 4)  # 45 degree rotation
scaling = scaling_matrix(2, 3)         # Scaling by 2x in x, 3x in y
translation = translation_matrix(1, 2) # Translate by (1, 2)

# Step 2: Create a target transformation (e.g., a combination of these with some noise)
target_matrix = 0.5 * rotation + 0.3 * scaling[:2, :2] + 0.2 * np.eye(2)  # A combination of rotation, scaling, and identity

# Step 3: Flatten matrices
A = np.vstack([rotation.flatten(), scaling[:2, :2].flatten(), np.eye(2).flatten()])  # Transformation matrices stacked
z = target_matrix.flatten()  # Target transformation matrix flattened

# Step 4: Solve for weights using regularized least squares
lambd = 1e-2  # Regularization parameter
# A_T_A = A.T @ A  # A^T A
A_T_A=A.transpose()*A
lambda_I = lambd * np.eye(A_T_A.shape[0])  # λI
A_T_z = A.transpose()*z # A^T z

# Solve for weights using the formula: (A^T A + λI)^(-1) A^T z
w = np.linalg.solve(A_T_A + lambda_I, A_T_z)

# Step 5: Reconstruct the matrix using the weights
reconstructed_matrix = w[0] * rotation + w[1] * scaling[:2, :2] + w[2] * np.eye(2)

# Display results
print("Weights:", w)
print("Target matrix:\n", target_matrix)
print("Reconstructed matrix:\n", reconstructed_matrix)
print("Difference:\n", target_matrix - reconstructed_matrix)
