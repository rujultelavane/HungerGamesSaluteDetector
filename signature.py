import cv2 as cv
import numpy as np
import streamlit as st

#read in
img = cv.imread("C:\\Users\\30004267\\PycharmProjects\\signature\\sigPic.jpg")
st.image(img, caption="Original Image", channels="BGR")

#grayscale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
st.image(gray_img, caption="Grayscale Image", channels="GRAY")

#remove bg
##CHANGE THRESHOLD
threshold_value = st.slider("Select Threshold Value", min_value=0, max_value=255, value=115, step=1)

_, trans_img = cv.threshold(gray_img, threshold_value, 255, cv.THRESH_BINARY)
st.image(trans_img, caption=f"Transparent Image (Threshold: {threshold_value})", channels="GRAY")

#save file
output_path = "C:\\Users\\30004267\\PycharmProjects\\signature\\TransparentImage.png"
cv.imwrite(output_path, trans_img)
#print(f"Image saved at {output_path}")

#transparent
alpha_channel = np.where(trans_img == 255, 0, 255).astype(np.uint8)
rgba_img = cv.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha_channel))
output_path = "C:\\Users\\30004267\\PycharmProjects\\signature\\TransparentFinalImage.png"
cv.imwrite(output_path, rgba_img)
#print(f"Image saved at {output_path}")

#gui - pick pen color
color = st.color_picker("Pick a color", "#000000")
color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) #convert hex to rbg

colored_img = np.zeros_like(img)
for i in range(3):
    colored_img[:, :, i] = np.where(trans_img == 0, color_rgb[i], img[:, :, i])

rgba_colored_img = cv.merge((*cv.split(colored_img), alpha_channel))
st.image(rgba_colored_img, caption="Colored Signature Image", channels="RGBA")

output_path = "C:\\Users\\30004267\\PycharmProjects\\signature\\TransparentColoredImage.png"
if st.button("Click to Save!"):
    cv.imwrite(output_path, rgba_colored_img)
    st.write(f"Image saved at {output_path}")


