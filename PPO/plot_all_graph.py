from PIL import Image

scale_list = ['5_5', '7_7', '10_10', '12_12', '15_15']

# 打开 SVG 图片
image1 = Image.open('PPO_figs/scale_5_5/PPO_scale_5_5_fig_0.jpg')
image2 = Image.open('PPO_figs/scale_7_7/PPO_scale_7_7_fig_0.jpg')
image3 = Image.open('PPO_figs/scale_10_10/PPO_scale_10_10_fig_0.jpg')
image4 = Image.open('PPO_figs/scale_12_12/PPO_scale_12_12_fig_0.jpg')
image5 = Image.open('PPO_figs/scale_15_15/PPO_scale_15_15_fig_0.jpg')

# 获取图片的大小
width1, height1 = image1.size
width2, height2 = image2.size
width3, height3 = image3.size
width4, height4 = image4.size
width5, height5 = image5.size

print(width1, height1)

total_width = width1 + width2
total_height = height1 + height3 + height5

# 创建一个新的空白图片
new_image = Image.new('RGB', (total_width, total_height), color='white')

# 将四张图片拼接到新图片上
new_image.paste(image1, (0, 0))
new_image.paste(image2, (width1, 0))
new_image.paste(image3, (0, height1))
new_image.paste(image4, (width1, height1))
new_image.paste(image5, (int(width1/2), height1 + height3))


# 保存拼接后的图片
new_image.save('concatenated_image.jpg')
