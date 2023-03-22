
# Display resulting triangular mesh using Matplotlib.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Task import verts,centres,faces,norm_S,norm_V

#mesh Z,X,Y co-ordinates of `verts`, `centres`, `norm_s`, and `norm_V`
Z, X, Y = zip(*verts)
z, x, y = zip(*centres)
sz,sx,sy = zip(*norm_S)
vz,vx,vy = zip(*norm_V)

fig = plt.figure(figsize=(50, 50))
ax = fig.add_subplot(111, projection='3d')

mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
mesh.set_facecolor('peachpuff')
ax.add_collection3d(mesh)
ax.quiver(Z, X, Y, vz, vx, vy, color='royalblue',arrow_length_ratio=0.15, alpha=0.5, label='Vertex Normals')
ax.quiver(z, x, y, sz, sx, sy, color='brown',arrow_length_ratio=0.4, linewidths=3.5, label='Surface Normals')

ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='y', labelsize=40)
ax.tick_params(axis='z', labelsize=40)

ax.set_ylim(bottom=20, top=60)

ax.set_xlabel("x-axis (mm)", fontsize=55, labelpad=40)
ax.set_ylabel("y-axis (mm)", fontsize=55, labelpad=40)
ax.set_zlabel("z-axis (mm)", fontsize=55, labelpad=40)

#set location and size of legend
ax.legend(loc='upper right', prop={'size':70})
plt.savefig('surface_vertex_vector_plots.png')


