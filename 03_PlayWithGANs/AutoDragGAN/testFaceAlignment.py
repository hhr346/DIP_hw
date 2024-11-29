'''
Test the library face_alignment
'''
import face_alignment
from skimage import io
import matplotlib.pyplot as plt

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

input_image = io.imread('../assets/ikun.jpg')
preds = fa.get_landmarks(input_image)[0]

print(preds[8])
print(preds[17])
print(preds[26])

# print('Plotting the landmarks on the input image...')
# fig, ax = plt.subplots()
# ax.imshow(input_image, aspect='equal')
# ax.axis('off')

# for point in preds:
#     ax.plot(point[0], point[1], 'ro', markersize=2)

# for point in preds:
#     ax.text(point[0] + 2, point[1] - 2, f'({point[0]:.0f}, {point[1]:.0f})', color='yellow', fontsize=8)

# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.savefig('./ikun_control.jpg', dpi=300)
