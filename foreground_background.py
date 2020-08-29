# Importing the necessary packages
import cv2
import numpy as np
import sys

live_vid = cv2.VideoCapture(0)     # Live video capturing (foreground)
bg_vid = cv2.VideoCapture(r"D:\python\project\Background\friends.mp4")   # Capturing the saved Video(background)

# Default width and height value
width = 800
height = 650

_, init_fg = live_vid.read()         # Initial background (for subtraction)
init_fg = cv2.resize(init_fg, (width, height))  # Resizing
if init_fg.all() != None:
    print("Background Detected")


# Infinite loop
try:
    while True:
        # Reading values from live cam and stored video
        _, new_fg = live_vid.read()        # reading the frames for foreground
        _, bg = bg_vid.read()       # reading the frames for background

        # Resizing for the matrix value to be same
        new_fg = cv2.resize(new_fg, (width, height))
        bg = cv2.resize(bg, (width, height))
        # -----cv2.imshow("initial", init_fg)-----

        sub_1 = cv2.subtract(new_fg, init_fg)       # performing both
        sub_2 = cv2.subtract(init_fg, new_fg)       # positive and negative subtraction
        tot_sub = sub_2 + sub_1
        tot_sub[abs(tot_sub < 14.5)] = 0     # attenuating the noises from the difference calculated

        # converting the result to gray for mask separation
        gray = cv2.cvtColor(tot_sub, cv2.COLOR_BGR2GRAY)
        gray[abs(gray) < 10] = 0    # For b& w separation
        fg_mask = gray.astype(np.uint8)     # converting to type = union-int-8bits
        fg_mask[fg_mask > 0] = 255      # Creating the b&w mask for foreground separation (white)
        # -----cv2.imshow("mask", fg_mask)-----
        bg_mask = cv2.bitwise_not(fg_mask)  # inverting fg mask to get bg mask (black)
        # -----cv2.imshow("inv", bg_mask)-----

        Fg = cv2.bitwise_and(new_fg, new_fg, mask=fg_mask)  # AND operation on colour(live video) by fg mask
        bg = cv2.bitwise_and(bg, bg, mask=bg_mask)  # AND operation on bg (saved video) by bg mask
        # -----cv2.imshow("fg", Fg)-----
        # -----cv2.imshow("bg", bg)-----
        dest = cv2.add(Fg, bg)      # adding fg bg to get the final fore(live) and back(saved) ground
        cv2.imshow("final", dest)   # Showing the result on window
        if cv2.waitKey(1) == 27:    # press-esc-quit
            break
    live_vid.release()
    cv2.destroyAllWindows()

except cv2.error as e:      # after the streaming of video 
    print(">>>>Exiting from the window")
    live_vid.release()
    cv2.destroyAllWindows()
    sys.exit(1)
