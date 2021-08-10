import sys

import trimesh as tm

from glinter.points.msms.msms_parser import read_msms

if __name__ == '__main__':
    vertices, faces, normalv, res_id = read_msms(sys.argv[1])
    mesh = tm.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True,
        validate=True,
    )
    ply_file = f'{sys.argv[1]}.ply'
    with open(ply_file, 'wb') as plyh:
        export = tm.exchange.ply.export_ply(
            mesh, encoding='ascii', vertex_normal=True,
        )
        plyh.write(export)
