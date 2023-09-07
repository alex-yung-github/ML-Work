import socket
import time
import imagezmq
import cv2

sender = imagezmq.ImageSender(connect_to='tcp://jeff-macbook:5555')
rpi_name = socket.gethostname() # send RPi hostname with each image
time.sleep(2.0)  # allow camera sensor to warm up

while True:  # send images as stream until Ctrl-C
    # image = picam.read()
    image = cv2.imread("download.png")
    sender.send_image(rpi_name, image)