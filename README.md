# Oil Palm Detection Model

## 1 - Pre-harvesting detection model

Classes: ripe fruit, underripe fruit, unripe fruit

## 2 - Post-harvesting detection model

Classes: abnormal, buah busuk, buah masak, buah lewat masak, buah mentah, tandan kosong

The detection model is a bit complicated since we need to tackle multiple issues in our images. There are 2 main issues; 1. Abnormal photos (low quality images, light exposure) and 2. Presence of "berondolan" objects or loose bunches. We have tackled using the following schematic. 

![image](https://github.com/yohanesnuwara/OilPalmVision/assets/51282928/3f6b9126-771b-4fc1-8cd7-48a25d5c9fcd)
