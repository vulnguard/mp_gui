from tkinter import filedialog
from image_util import *

from tkinter import Label, Button, Frame
import tkinter
import cv2

import PIL.Image, PIL.ImageTk

import matplotlib as mpl

class MyGui:
    def __init__(self, window, window_title):
        self.window = window
        self.choice_window = None

        self.window.title(window_title)
        self.image_history = []
        self.undo_history = []
        self.did_undo = False
        self.reset_redo = False

        self.label = Label(self.window, text="Select an option to play with images.", bg = "magenta", font='Helvetica 18 bold')
        self.label.grid(row = 0)

        self.button_frame = Frame(window, bg="blue")
        self.button_frame.grid(row = 1)

        self.init_buttons()

        # Setup a frame for the iamges
        self.image_frame = Frame(self.window, bg="white")
        self.image_frame.grid(row = 2)

        # Set up the labels for the images.
        self.origional_image_label = tkinter.Label(self.image_frame, text="Origional (0x0)")
        self.origional_image_label.grid(row = 1, column = 0)

        self.previous_image_label = tkinter.Label(self.image_frame, text="Previous (0x0)")
        self.previous_image_label.grid(row = 1, column = 1)

        self.current_image_label = tkinter.Label(self.image_frame, text="Current (0x0)")
        self.current_image_label.grid(row = 1, column = 2)

        # Create a canvas for the images.
        self.origional_image_canvas = tkinter.Canvas(self.image_frame, width = 200, height = 200)
        self.origional_image_canvas.grid(row = 2, column = 0)

        self.previous_image_canvas = tkinter.Canvas(self.image_frame, width = 200, height = 200)
        self.previous_image_canvas.grid(row = 2, column = 1)

        self.current_image_canvas = tkinter.Canvas(self.image_frame, width = 200, height = 200)
        self.current_image_canvas.grid(row = 2, column = 2)

    def close_window(self):
        self.window.quit()

    def update_canvas(self):
        origional_image = self.image_history[0]
        modded_image = self.image_history[self.curr_index]
        if self.curr_index > 0:
            prev_image = self.image_history[self.curr_index - 1]
        else:
            prev_image = origional_image

        rows, cols = get_image_dimentions(origional_image)
        self.origional_image_canvas.config(width = cols, height = rows)
        self.origional_image_label.config(text="Origional ({}x{})".format(rows, cols))

        rows, cols = get_image_dimentions(prev_image)
        self.previous_image_canvas.config(width = cols, height = rows)
        self.previous_image_label.config(text="Previous ({}x{})".format(rows, cols))

        rows, cols = get_image_dimentions(modded_image)
        self.current_image_canvas.config(width = cols, height = rows)
        self.current_image_label.config(text="Current ({}x{})".format(rows, cols))

    def load_image(self):
        path = filedialog.askopenfilename(initialdir = "res/prac2")

        print("Loading file...")
        if not os.path.isfile(path):
            print("Error loading image.")
            return

        self.image_history = []
        self.curr_index = 0
        self.image_history.append(cv2.cvtColor(load_image(path), cv2.COLOR_BGR2RGB))

        self.image_name = path.split("/")[-1]
        print("Loaded image '{}'.".format(self.image_name))

        self.update_canvas()
        self.draw_current_image()
        self.draw_modified_image()

    def draw_current_image(self):
        self.curr_main_image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.image_history[0]))
        self.current_image_canvas.delete("all")
        self.origional_image_canvas.create_image(0, 0, image=self.curr_main_image, anchor=tkinter.NW)

    def draw_modified_image(self):
        self.curr_mod_image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.image_history[self.curr_index]))
        self.current_image_canvas.create_image(0, 0, image=self.curr_mod_image, anchor=tkinter.NW)
        if self.curr_index == 1 or self.curr_index == 0:
            self.prev_mod_image = self.curr_main_image
        else:
            self.prev_mod_image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.image_history[self.curr_index - 1]))

        self.previous_image_canvas.delete("all")
        self.previous_image_canvas.create_image(0, 0, image=self.prev_mod_image, anchor=tkinter.NW)

    def update_modded_image(self, modded_image):
        self.image_history.append(modded_image)
        self.curr_index += 1
        self.update_canvas()
        self.draw_modified_image()

    def has_image(self):
        return len(self.image_history) != 0

    # row 0
    def save_image(self):
        if not self.has_image():
            print("Cannot save empty image.")
            return

        file = filedialog.asksaveasfile(mode='w', defaultextension=".png")
        if file:
            save_image(bgr_to_rgb(self.image_history[-1]), file.name)

    def notify_mod_operation(self, new_image):
        if self.did_undo:
            self.did_undo = False
            self.undo_history = []

        self.update_modded_image(new_image)


    # Modifiers (row1)
    def convolve(self):
        if not self.has_image():
            print("Cannot convolve empty image.")
            return

        choices = {\
                   "perwit_kx": perwit_kx_kernal(),
                   "perwit_ky": perwit_ky_kernal(),\
                   "sobel_kx": sobel_kx_kernal(),\
                   "sobel_ky": sobel_ky_kernal(),\
                   "laplacian": laplacian_kernal(),\
                   "gaussian": gaussian_kernal()
                  }

        self.get_choice_from_buttons(choices, callback = self.do_conv)

    def do_conv(self):
        kernel = self.curr_choice

        modded = apply_convolution(self.image_history[self.curr_index], kernel)
        self.notify_mod_operation(modded)

    def get_choice_from_buttons(self, choices, callback):
        if self.choice_window is not None:
            self.choice_window.destroy()

        self.choice_window = tkinter.Toplevel(self.window)
        self.choice_window.wm_title("Choose an option")
        for key in choices.keys():
            self.add_button(self.choice_window, key, lambda choice=choices[key]: self.set_choice(choice, callback))


    def get_choice_from_values(self, label_texts, default_values, callback):
        if self.choice_window is not None:
            self.choice_window.destroy()

        self.choice_window = tkinter.Toplevel(self.window)
        self.choice_window.wm_title("Enter values")

        self.input_box_labels = []
        self.input_boxes = []
        index = 0
        for label in label_texts:
            self.input_box_labels.append(tkinter.Label(self.choice_window, text=label, width = 10))
            self.input_box_labels[index].grid(row = 0, column = index)

            self.input_boxes.append(tkinter.Entry(self.choice_window, width = 10))
            self.input_boxes[index].insert(0, default_values[index])
            self.input_boxes[index].grid(row = 1, column = index)
            index += 1

        self.add_button(self.choice_window, "Done.", row = 2, col = index,\
                        command = lambda: self.set_choice_from_entrys(callback))

    def set_choice_from_entrys(self, callback):
        self.curr_choice = []
        for input_box in self.input_boxes:
            self.curr_choice.append(input_box.get())

        self.choice_window.destroy()
        callback()

    def set_choice(self, value, callback):
        self.curr_choice = value

        self.choice_window.destroy()
        callback()

    def gaussian(self):
        if not self.has_image():
            print("Cannot gauss empty image.")
            return

        modded = general_gaussian_filter(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def median(self):
        if not self.has_image():
            print("Cannot median empty image.")
            return

        modded = apply_median_filter(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def norm_hist_bw(self):
        if not self.has_image():
            print("Cannot norm(bw) empty image.")
            return

        modded = normalize_histogram_bnw(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def norm_hist_col(self):
        if not self.has_image():
            print("Cannot norm(col) empty image.")
            return

        modded = normalize_histogram_bgr(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)


    # Modifiers (row2)
    def crop(self):
        if not self.has_image():
            print("Cannot crop empty image.")
            return

        defaults = [0, 0, 100, 100]
        self.get_choice_from_values(["start_x", "start_y", "width", "height"], defaults, self.do_crop)

    def do_crop(self):
        vals = self.curr_choice
        # x1, y1, x2, y2
        x = int(vals[0])
        y = int(vals[1])
        width = int(vals[2])
        height = int(vals[3])

        x1 = x + width
        y1 = y + height

        image_height, image_width = get_image_dimentions(self.image_history[self.curr_index])

        if x > x1 or y > y1 or x1 > image_width or y1 > image_height:
            print("Invalid crop dimentions.")
            return

        modded = crop_image(self.image_history[self.curr_index], x, y, width, height)
        self.update_canvas()
        self.notify_mod_operation(modded)

    def rotate(self):
        if not self.has_image():
            print("Cannot rotate empty image.")
            return

        defaults = [45]
        self.get_choice_from_values(["Angle (deg)"], defaults, self.do_rotate)

    def do_rotate(self):
        angle = int(self.curr_choice[0])

        modded = rotate_image(self.image_history[self.curr_index], angle)
        self.update_canvas()
        self.notify_mod_operation(modded)

    def resize(self):
        if not self.has_image():
            print("Cannot resize empty image.")
            return

        defaults = [0.5, 0.5]
        self.get_choice_from_values(["Factor (x)", "Factor (y)"], defaults, self.do_resize)

    def do_resize(self):
        x_factor = float(self.curr_choice[0])
        y_factor = float(self.curr_choice[1])

        modded = resize_image(self.image_history[self.curr_index], x_factor, y_factor)
        self.update_canvas()
        self.notify_mod_operation(modded)

    def affine_transform(self):
        if not self.has_image():
            print("Cannot affine_transform empty image.")
            return

        defaults = ["50,50", "200,50", "50,200", "10,100", "200,50", "100,250"]
        self.get_choice_from_values(["start_p1", "start_p2", "start_p3",\
                                     "end_p1", "end_p2", "end_p3"], defaults, self.do_affine)

    def do_affine(self):
        values = self.curr_choice
        all_points = []
        for point in values:
            point = point.split(",")
            all_points.append([int(point[0]), int(point[1])])

        modded = apply_affine_transformation(self.image_history[self.curr_index], all_points[0:3], all_points[3:6])
        self.update_canvas()
        self.notify_mod_operation(modded)


    def pixel_transform(self):
        if not self.has_image():
            print("Cannot pixel_transform empty image.")
            return

        defaults = [0.5, 10]
        self.get_choice_from_values(["Alpha", "Beta"], defaults, self.do_aff_pixel)

    def do_aff_pixel(self):
        alpha = float(self.curr_choice[0])
        beta = float(self.curr_choice[1])

        modded = apply_affine_pixel_transform(self.image_history[self.curr_index], alpha, beta)
        self.notify_mod_operation(modded)

    def draw_rect(self):
        if not self.has_image():
            print("Cannot draw_rect on empty image.")
            return

        defaults = [30, 30, 50, 50, 2, "0, 0, 0"]
        self.get_choice_from_values(["start_x", "start_y", "width", "height", "thickness", "colour(r,g,b)"], defaults, self.do_draw_rect)

    def do_draw_rect(self):
        vals = self.curr_choice
        # x1, y1, x2, y2
        x = int(vals[0])
        y = int(vals[1])
        width = int(vals[2])
        height = int(vals[3])

        x1 = x + width
        y1 = y + height

        thickness = int(vals[4])
        rgb_vals = vals[5].split(",")
        color = (int(rgb_vals[0]), int(rgb_vals[1]), int(rgb_vals[2]))

        image_height, image_width = get_image_dimentions(self.image_history[self.curr_index])

        if x > x1 or y > y1 or x1 > image_width or y1 > image_height:
            print("Invalid rectangle dimentions.")
            return

        start = (x, y)
        end = (x + width, y + height)
        modded = draw_rect_on_image(self.image_history[self.curr_index], start, end, thickness, color)
        self.update_canvas()
        self.notify_mod_operation(modded)

    # Modifiers (row3)
    def inverse_bw(self):
        if not self.has_image():
            print("Cannot inverse_bw of empty image.")
            return

        modded = inverse_bw(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)


    # Morphological
    def m_dilate(self):
        if not self.has_image():
            print("Cannot dilate empty image.")
            return

        modded = morph_dilate(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def m_erode(self):
        if not self.has_image():
            print("Cannot erode empty image.")
            return

        modded = morph_erode(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def m_open(self):
        if not self.has_image():
            print("Cannot open empty image.")
            return

        modded = morph_open(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def m_close(self):
        if not self.has_image():
            print("Cannot close empty image.")
            return

        modded = morph_close(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def m_grad(self):
        if not self.has_image():
            print("Cannot grad empty image.")
            return

        modded = morph_gradiant(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def m_blackhat(self):
        if not self.has_image():
            print("Cannot blackhat empty image.")
            return

        modded = morph_blackhat(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    # Modifiers (row4)
    def to_rgb(self):
        if not self.has_image():
            print("Cannot convert empty image.")
            return

        modded = bgr_to_rgb(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def to_bgr(self):
        if not self.has_image():
            print("Cannot convert empty image.")
            return

        modded = rgb_to_bgr(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def to_gray(self):
        if not self.has_image():
            print("Cannot convert empty image.")
            return

        modded = to_grayscale(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def to_hsv(self):
        if not self.has_image():
            print("Cannot convert empty image.")
            return

        modded = to_hsv(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def to_luv(self):
        if not self.has_image():
            print("Cannot convert empty image.")
            return

        modded = to_luv(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def to_lab(self):
        if not self.has_image():
            print("Cannot convert empty image.")
            return

        modded = to_lab(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def to_yuv(self):
        if not self.has_image():
            print("Cannot convert empty image.")
            return

        modded = to_yuv(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    # Other
    def undo(self):
        if len(self.image_history) == 1 or len(self.image_history) == 0:
            print("Cannot undo no-op.")
            return

        self.did_undo = True

        self.undo_history.append(self.image_history.pop())
        self.curr_index -= 1
        self.update_canvas()
        self.draw_modified_image()

    def redo(self):
        if len(self.undo_history) == 0:
            print("Cannot redo no-op.")
            return

        self.image_history.append(self.undo_history.pop())
        self.curr_index += 1
        self.update_canvas()
        self.draw_modified_image()

    def reset(self):
        if len(self.image_history) == 1 or len(self.image_history) == 0:
            print("Cannot reset nothing.")
            return

        first_image = self.image_history[0]
        self.image_history = []
        self.undo_history = []
        self.did_undo = False
        self.reset_redo = False

        self.image_history.append(first_image)
        self.curr_index = 0
        self.update_canvas()
        self.draw_modified_image()

    def no_op(self):
        modded = self.image_history[self.curr_index].copy()
        self.notify_mod_operation(modded)

    # Init stuff
    def init_buttons(self):
        self.buttons = []
        self.curr_button_row = 0
        self.curr_button_col = 2

        self.add_button(self.button_frame, "Load image", self.load_image, "green", 0)

        self.curr_button_col = 4
        self.add_button(self.button_frame, "Save image", self.save_image, "green", 0)

        # Modifiers (row1)
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.button_frame, "convolve", self.convolve)
        self.add_button(self.button_frame, "gaussian", self.gaussian)
        self.add_button(self.button_frame, "med-filter", self.median)
        self.add_button(self.button_frame, "norm_bw", self.norm_hist_bw)
        self.add_button(self.button_frame, "norm_col", self.norm_hist_col)

        # Modifiers (row2)
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.button_frame, "crop", self.crop)
        self.add_button(self.button_frame, "rotate", self.rotate)
        self.add_button(self.button_frame, "resize", self.resize)
        self.add_button(self.button_frame, "affine", self.affine_transform)
        self.add_button(self.button_frame, "affine pixel", self.pixel_transform)
        self.add_button(self.button_frame, "draw rect", self.draw_rect)

        # Modifiers (row3)
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.button_frame, "inverse_bw", self.inverse_bw)

        # Morphological
        self.add_button(self.button_frame, "m_dilate", self.m_dilate)
        self.add_button(self.button_frame, "m_erode", self.m_erode)
        self.add_button(self.button_frame, "m_open", self.m_open)
        self.add_button(self.button_frame, "m_close", self.m_close)
        self.add_button(self.button_frame, "m_grad", self.m_grad)
        self.add_button(self.button_frame, "m_blackhat", self.m_blackhat)

        # Modifiers (row4)
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.button_frame, "to_rgb", self.to_rgb)
        self.add_button(self.button_frame, "to_gray", self.to_gray)
        self.add_button(self.button_frame, "to_bgr", self.to_bgr)
        self.add_button(self.button_frame, "to_hsv", self.to_hsv)
        self.add_button(self.button_frame, "to_luv", self.to_luv)
        self.add_button(self.button_frame, "to_lab", self.to_lab)
        self.add_button(self.button_frame, "to_yuv", self.to_yuv)

        # No-op
        self.add_button(self.button_frame, "no-op", self.no_op, "yellow", row = 0, col = 6)

        # Reset
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.button_frame, "Reset", self.reset, "orange")

        # Undo
        self.curr_button_col = 3
        self.add_button(self.button_frame, "Undo", self.undo, "yellow")

        # Redo
        self.curr_button_col = 3
        self.curr_button_col = 4
        self.add_button(self.button_frame, "Redo", self.redo, "yellow")

        # Close
        self.curr_button_col = 6
        self.add_button(self.button_frame, "Close", self.close_window, "red")

    def add_button(self, frame, text, command, background = "gray", row = None, col = None):
        if row is None:
            row = self.curr_button_row
        if col is None:
            col = self.curr_button_col
            self.curr_button_col += 1

        # print("adding button '{}' to {}, {}".format(text, row, col))
        self.buttons.append(Button(frame, text=text, command=command, bg = background))
        self.buttons[-1].grid(row = row, column = col)
