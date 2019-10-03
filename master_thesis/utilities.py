import numpy as np

def distance(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=1))

def Rotate(Alpha, Beta, Gamma, VECTOR):
    Alpha = Alpha * np.pi/180; Beta = Beta * np.pi/180; Gamma = Gamma * np.pi/180

    Rotation_matrix_X = np.array([[1, 0, 0],[0, np.cos(Alpha), -np.sin(Alpha)], [0, np.sin(Alpha), np.cos(Alpha)]])
    Rotation_matrix_Y = np.array([[np.cos(Beta), 0, np.sin(Beta)],[0, 1, 0], [-np.sin(Beta), 0, np.cos(Beta)]])
    Rotation_matrix_Z = np.array([[np.cos(Gamma), -np.sin(Gamma), 0],[np.sin(Gamma), np.cos(Gamma), 0], [0,0,1]])
    
    rotation_around_X = np.matmul(Rotation_matrix_X, VECTOR.T)
    rotation_around_XY = np.matmul(Rotation_matrix_Y, rotation_around_X)
    rotation_around_XYZ = np.matmul(Rotation_matrix_Z, rotation_around_XY)
    VECTOR = np.transpose(rotation_around_XYZ).astype(np.float32)
    return VECTOR

# data.shape (molecules, atoms, positions)
def data_to_gro(gro_filename, data, system_name, molecule_type, box):
    molecules_in_system = data.shape[0]
    atoms_in_molecule = data.shape[1]
    n_atoms = molecules_in_system * atoms_in_molecule

    index = 0
    with open(gro_filename+'.gro', 'w') as f:
        f.write(system_name)
        f.write('\n %s \n' % n_atoms)

        for i, position in enumerate(data):
            for j in range(atoms_in_molecule):
                index += 1
                numbers = ("%.3f" % position[j][0]).rjust(8) + ("%.3f" % position[j][1]).rjust(8) + ("%.3f" % position[j][2]).rjust(8)
                f.write((str(i+1)+molecule_type).rjust(10) + ' atom' + str(index).rjust(5) + numbers + '\n')


        string = '   '
        for element in box:
            string += str("%.3f" % element) + ' '

        f.write(string[:-1]+'\n')