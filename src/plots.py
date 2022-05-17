import matplotlib.pyplot as plt
import numpy as np

dura_red = [0.87, 0.848, 0.835, 0.841, 0.789, 0.816, 0.75, 0.925, 0.914, 0.826, 0.833]
dura_green = [0.487, 0.432, 0.432, 0.466, 0.411, 0.388, 0.405, 0.299, 0.341, 0.252, 0.56]
dura_blue = [0.603, 0.538, 0.522, 0.565, 0.498, 0.477, 0.478, 0.413, 0.449, 0.468, 0.624]

bone_red = [0.913, 0.937, 0.939, 0.944, 0.94, 0.877, 0.923, 0.929, 0.933, 0.85, 0.678]
bone_green = [0.609, 0.674, 0.657, 0.662, 0.404, 0.436, 0.399, 0.399, 0.398, 0.833, 0.524]
bone_blue = [0.696, 0.742, 0.726, 0.734, 0.488, 0.504, 0.513, 0.481, 0.479, 0.849, 0.571]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(dura_red, dura_green, dura_blue, marker='o', label='dura')
ax.scatter(bone_red, bone_green, bone_blue, marker='^', label='bone')

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

ax.legend()
plt.show()
