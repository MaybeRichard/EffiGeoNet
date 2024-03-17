# import os
#
# # 图片所在的文件夹路径
# folder_path = '../dataset/fixed'
#
# # 从文件夹获取所有图片
# image_files = os.listdir(folder_path)
#
# # 遍历图片文件
# for image_file in image_files:
#     # 检查文件名是否以 .png 结尾
#     if image_file.endswith('.png'):
#         # 去掉 .png 后缀，获取数字部分
#         number = int(image_file[:-4])-1
#
#         # 使用 zfill() 方法将数字部分补齐为6位数，并加上 .png 后缀
#         new_image_file = str(number).zfill(6) + '.png'
#
#         # 获取图片原始路径和新路径
#         original_path = os.path.join(folder_path, image_file)
#         new_path = os.path.join(folder_path, new_image_file)
#
#         # 重命名图片文件
#         os.rename(original_path, new_path)
#
# print("图片文件名修改完成！")


import os
from PIL import Image
#
# # 图片所在的文件夹路径
# folder_path = '../dataset/fixed'
#
# # 目标尺寸
# target_size = (512, 512)
#
# # 获取文件夹中的所有文件
# image_files = os.listdir(folder_path)
#
# # 遍历文件夹中的所有文件
# for image_file in image_files:
#     # 检查文件是否是图片
#     if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#         # 获取图片的完整路径
#         image_path = os.path.join(folder_path, image_file)
#
#         # 打开图片并调整尺寸
#         img = Image.open(image_path)
#         resized_img = img.resize(target_size, Image.ANTIALIAS)
#
#         # 保存调整尺寸后的图片（覆盖原图片）
#         resized_img.save(image_path)
#
# print("图片尺寸调整完成！")

img = Image.open("../dataset/fixed-1024/000001.png")
img = img.resize((1024, 1024))
img.save("../dataset/fixed-1024/000001.png")