import numpy as np
import trimesh as tm

def read_vertices(path, resolution=2, require_connected=True):
    with open(path, 'rb') as mh:
        mesh = tm.Trimesh(**tm.exchange.ply.load_ply(mh))
        print(mesh)

    return vert

def read_msms(vert_path, face_path):
    nv = None
    verts = []
    with open(vert_path, 'rt') as h:
        for l in h:
            l = l.strip()
            if not l or l.startswith('#'):
                continue
            fields = l.split()
            if nv is None: # read the header
                nv = int(fields[0])
                continue
            verts.append([float(_) for _ in fields[:6]])
    verts = np.array(verts)
    coords, normals = verts[:,:3], verts[:,3:]
    
    assert coords.shape[0] == nv

    nf = None
    faces = []
    with open(face_path, 'rt') as h:
        for l in h:
            l = l.strip()
            if not l or l.startswith('#'):
                continue
            fields = l.split()
            if nf is None:
                nf = int(fields[0])
                continue
            faces.append([int(_) for _ in fields[:3]])
    faces = np.array(faces, dtype=np.long) - 1# MSMS vertex id starts from 1

    assert faces.shape[0] == nf

    return coords, faces, normals

def sample_points(
    verts, faces, normals, require_connected=False, resolution=0.8
):
    """
    resolution (float, 0.8) : 
        empirically, the number of vertices starts to plateau after 0.8 
    """
    mesh = tm.Trimesh(
        vertices=verts, faces=faces, vertex_normals=normals, validate=True,
        process=True,
    )
    # reweight vertex normals
    normals = mesh.vertex_normals
    if require_connected:
        # find the largest connected component
        edge = mesh.edges_unique
        cc = tm.graph.connected_components(edge)
        _lengths = [len(c) for c in cc]
        i = np.argmax(_lengths)
        if _lengths[i] / np.sum(_lengths) < 0.95:
            print(
                f'{path} is **significantly** disconnected'
            )
            return
        verts = mesh.vertices[cc[i]]
        normals = normals[cc[i]]
    else:
        verts = mesh.vertices

    if resolution > 0:
        verts, selected = tm.points.remove_close(verts, resolution)
        normals = mesh.vertex_normals[selected]
    
    return verts, normals

def plot_mesh(mesh):
    """
    use mplot3d to plot meshes, try not to render big meshes
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = mesh.vertices.T
    ax.plot_trisurf(x, y, z, triangles=mesh.faces)
    plt.show()

def plot_normals(verts, normals, magnify=1, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
    # Make the grid
    # x, y, z = np.meshgrid(
    #     np.arange(-0.8, 1, 0.2),
    #     np.arange(-0.8, 1, 0.2),
    #     np.arange(-0.8, 1, 0.8),
    # )
    x, y, z= verts.T 
    normals = normals * magnify
    u, v, w= normals.T 
    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=False)
    plt.show()
 
if __name__ == '__main__':
    from pathlib import Path
    import sys
    import matplotlib.pyplot as plt
    verts, faces, normals = read_msms(Path(sys.argv[1]), Path(sys.argv[2]))
    _verts, _normals = sample_points(verts, faces, normals, resolution=0.8)

    # visual normal vectors
    # plot_normals(verts, normals, magnify=10)

    # visual resolution vs number of vertices plots
    # _lens = []
    # resolutions = np.arange(0, 1.001, 0.025)
    # for res in resolutions:
    #     _verts, _normals = sample_points(
    #         verts, faces, normals, resolution=float(res),
    #     )
    #     _lens.append(len(_verts))

    # plt.plot(resolutions, _lens)
    # plt.show()
