from feature_detection import *
from image_processing import *

from tkinter import Label, Button, Frame, filedialog
import tkinter
import cv2

import PIL.Image, PIL.ImageTk
import matplotlib as mpl

MAX_ROWS_ORIGIONAL = 300
MAX_COLS_ORIGIONAL = 600

class MyGui:
    def __init__(self, window, window_title):
        self.window = window
        self.main_frame = Frame(self.window, bg="white")
        self.choice_window = None

        self.window.title(window_title)
        self.image_history = []
        self.undo_history = []
        self.buttons = []
        self.spacer_frames = []
        self.saved_image = [None, None, None]
        self.saved_image_arrays = [None, None, None]
        self.did_undo = False
        self.reset_redo = False

        self.label = Label(self.main_frame, text="Select an option to play with images.", bg = "#4296c4", font='Helvetica 18 bold')
        self.label.grid(row = 0)

        # Setup button frames
        self.top_most_frame = Frame(self.main_frame, bg="white")
        self.top_most_frame.grid(row = 1)

        self.left_button_frame = Frame(self.top_most_frame, bg="blue")
        self.left_button_frame.grid(row = 0, column = 0)

        self.spacer_frames.append(Frame(self.top_most_frame, width = 20, bg="white"))
        self.spacer_frames[-1].grid(row = 0, column = 1)

        self.right_button_frame = Frame(self.top_most_frame, width = 150, bg="blue")
        self.right_button_frame.grid(row = 0, column = 2)

        self.other_button_frame = Frame(self.main_frame, bg="#cc1ec0")
        self.other_button_frame.grid(row = 2)

        self.init_left_buttons()
        self.init_right_buttons()
        self.init_other_buttons()

        # Setup a frame for the images
        self.image_frame = Frame(self.main_frame, bg="white")
        self.image_frame.grid(row = 3)

        self.save_frame = Frame(self.main_frame, bg="white")
        self.save_frame.grid(row = 4)

        self.sl_frame0 = Frame(self.save_frame, bg="white")
        self.sl_frame0.grid(column = 0, row = 1)
        self.sl_frame1 = Frame(self.save_frame, bg="white")
        self.sl_frame1.grid(column = 1, row = 1)
        self.sl_frame2 = Frame(self.save_frame, bg="white")
        self.sl_frame2.grid(column = 2, row = 1)

        self.init_save_laod_buttons()

        # Set up the labels for the images.
        self.origional_image_label = tkinter.Label(self.image_frame, text="Origional (Not to scale)\nOrigional dims: (0x0)")
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

        # Setup frames for saved images.
        self.save_canvas = []
        self.save_canvas.append(tkinter.Canvas(self.save_frame, width = 200, height = 200))
        self.save_canvas[0].grid(row = 0, column = 0)

        self.save_canvas.append(tkinter.Canvas(self.save_frame, width = 200, height = 200))
        self.save_canvas[1].grid(row = 0, column = 1)

        self.save_canvas.append(tkinter.Canvas(self.save_frame, width = 200, height = 200))
        self.save_canvas[2].grid(row = 0, column = 2)

        self.main_frame.pack()

    def close_window(self):
        self.get_choice_from_buttons({"Confirm Exit": True, "Don't Exit": False}, callback = self.do_exit)

    def do_exit(self):
        if not self.curr_choice:
            return

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
        label_text = "Origional ({}{}x{}){}"\
                     .format("Rescaled to: " if self.scaled_to_fit else "", rows, cols,\
                     "\nOrigional dims: ({}x{})".format(self.origional_rows, self.origional_cols)\
                     if self.scaled_to_fit else "")
        self.origional_image_label.config(text=label_text)

        rows, cols = get_image_dimentions(prev_image)
        self.previous_image_canvas.config(width = cols, height = rows)
        self.previous_image_label.config(text="Previous ({}x{})".format(rows, cols))

        rows, cols = get_image_dimentions(modded_image)
        self.current_image_canvas.config(width = cols, height = rows)
        self.current_image_label.config(text="Current ({}x{})".format(rows, cols))

    def ensure_image_fit(self):
        origional_image = self.image_history[0]
        self.origional_rows, self.origional_cols = get_image_dimentions(origional_image)

        # Rescale oriigonal if its too big so that it looks nicer.
        self.scaled_to_fit = False

        rows, cols = self.origional_rows, self.origional_cols

        row_factor = col_factor = 1
        if rows > MAX_ROWS_ORIGIONAL:
            row_factor = MAX_ROWS_ORIGIONAL / rows
        if cols > MAX_COLS_ORIGIONAL:
            col_factor = MAX_COLS_ORIGIONAL / cols

        min_factor = min(row_factor, col_factor)

        if not min_factor == 1:
            self.scaled_to_fit = True

        self.image_history[0] = resize_image(self.image_history[0], min_factor, min_factor)

    def load_image(self):
        self.resized_to_fit = False
        path = filedialog.askopenfilename(initialdir = "res/prac3")

        print("Loading file...")
        if not os.path.isfile(path):
            print("Error loading image... given path not a file.")
            return

        self.image_history = []
        self.curr_index = 0
        self.image_history.append(cv2.cvtColor(load_image(path), cv2.COLOR_BGR2RGB))

        self.ensure_image_fit()

        self.image_name = path.split("/")[-1]
        print("Loaded image '{}'.".format(self.image_name))

        self.update_canvas()
        self.draw_origional_image()
        self.draw_modified_image()

    def draw_origional_image(self):
        self.origional_image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.image_history[0]))
        self.origional_image_canvas.delete("all")
        self.origional_image_canvas.create_image(0, 0, image=self.origional_image, anchor=tkinter.NW)

    def draw_modified_image(self):
        self.curr_mod_image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.image_history[self.curr_index]))
        self.current_image_canvas.create_image(0, 0, image=self.curr_mod_image, anchor=tkinter.NW)
        if self.curr_index == 1 or self.curr_index == 0:
            self.prev_mod_image = self.origional_image
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

    def perspecrive_transform(self):
        if not self.has_image():
            print("Cannot perspecrive_transform empty image.")
            return

        image_height, image_width = get_image_dimentions(self.image_history[self.curr_index])
        max_tl = ["{}, {}".format(0, 0)]
        max_tr = ["{}, {}".format(image_width, 0)]
        max_bl = ["{}, {}".format(0, image_height)]
        max_br = ["{}, {}".format(image_width, image_height)]
        maxes = max_tl + max_tr + max_bl + max_br

        defaults = ["20, 20"] + ["280, 30"] + ["10, 210"] + ["210, 188"] + maxes
        self.get_choice_from_values(["s_topleft", "s_topright", "s_botleft", "s_botright",\
                                     "e_topleft", "e_topright", "e_botleft", "e_botright"],\
                                     defaults, self.do_perspective)

    def do_perspective(self):
        start_points = []
        end_points = []
        values = self.curr_choice

        num_vals = len(values)
        for ii in range(0, num_vals):
            x, y = values[ii].strip().split(",")
            int_val = [int(x), int(y)]
            if ii < num_vals / 2:
                start_points.append(int_val)
            else:
                end_points.append(int_val)

        points = start_points + end_points
        image_height, image_width = get_image_dimentions(self.image_history[self.curr_index])

        if len(start_points) != len(end_points) or len(start_points) != 4:
            print("Invalid number of points entered.")
            return

        for x, y in points:
            if x > image_width or y > image_height:
                print("Invalid point entered.")
                return

        modded = apply_perspective_transformation(self.image_history[self.curr_index], start_points, end_points)
        self.update_canvas()
        self.notify_mod_operation(modded)


    def affine_transform(self):
        if not self.has_image():
            print("Cannot affine_transform empty image.")
            return

        defaults = ["50, 50"] + ["200, 50"] + ["50, 200"] + ["10, 100"] + ["200, 50"] + ["10, 250"]
        self.get_choice_from_values(["start_p1", "start_p2", "start_p3",\
                                     "end_p1", "end_p2", "end_p3"],\
                                     defaults, self.do_affine)

    def do_affine(self):
        start_points = []
        end_points = []
        values = self.curr_choice

        num_vals = len(values)

        for ii in range(0, num_vals):
            x, y = values[ii].strip().split(",")
            int_val = [int(x), int(y)]
            if ii < num_vals / 2:
                start_points.append(int_val)
            else:
                end_points.append(int_val)

        points = start_points + end_points
        image_height, image_width = get_image_dimentions(self.image_history[self.curr_index])

        if len(start_points) != len(end_points) or len(start_points) < 2:
            print("Invalid number of points entered.")
            return

        for x, y in points:
            if x > image_width or y > image_height:
                print("Invalid point entered.")
                return

        modded = apply_affine_transformation(self.image_history[self.curr_index], start_points, end_points)
        self.update_canvas()
        self.notify_mod_operation(modded)



    def pixel_transform(self):
        if not self.has_image():
            print("Cannot pixel_transform empty image.")
            return

        modded = apply_affine_pixel_transform(self.image_history[self.curr_index], 1, 1)
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
        if not self.has_image():
            print("Empty no-op has EXTRA no effect.")
            return

        modded = self.image_history[self.curr_index].copy()
        self.notify_mod_operation(modded)


    # Corners
    def d_corners_st(self):
        if not self.has_image():
            print("Cannot detect corners on empty image.")
            return

        defaults = [25, 0.01, 10, "255, 0, 0"]
        self.get_choice_from_values(["num_corners", "quality", "distance", "colour(r,g,b)"], defaults, self.do_corners_st)


    def do_corners_st(self):
        vals = self.curr_choice
        num_corners = int(vals[0])
        quality = float(vals[1])

        dist = int(vals[2])
        rgb_vals = vals[3].split(",")
        color = (int(rgb_vals[0]), int(rgb_vals[1]), int(rgb_vals[2]))

        modded = detect_corners_st(self.image_history[self.curr_index], num_corners, quality, dist, color)
        self.notify_mod_operation(modded)


    def d_corners_h(self):
        if not self.has_image():
            print("Cannot detect corners on empty image.")
            return

        defaults = [0.01, 2, 3, 0.04, "255, 0, 0"]
        self.get_choice_from_values(["threshold", "blockSize", "ksize", "k", "colour(r,g,b)"], defaults, self.do_corners_h)


    def do_corners_h(self):
        vals = self.curr_choice
        threshold = float(vals[0])
        blockSize = int(vals[1])
        ksize = int(vals[2])
        k = float(vals[3])
        rgb_vals = vals[4].split(",")
        color = (int(rgb_vals[0]), int(rgb_vals[1]), int(rgb_vals[2]))

        modded = detect_corners_h(self.image_history[self.curr_index], threshold, blockSize, ksize, k, color)
        self.notify_mod_operation(modded)

    # Edges
    def d_edges_manual(self):
        if not self.has_image():
            print("Cannot detect edges on empty image.")
            return

        modded = detect_edges_manual(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def d_edges_auto(self):
        if not self.has_image():
            print("Cannot detect edges on empty image.")
            return

        modded = detect_edges_auto(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    # Lines
    def d_lines_manual(self):
        if not self.has_image():
            print("Cannot detect lines on empty image.")
            return

        modded = detect_lines_manual(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    def d_lines_auto(self):
        if not self.has_image():
            print("Cannot detect lines on empty image.")
            return

        modded = detect_lines_auto(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    # Blobs
    def d_blobs(self):
        if not self.has_image():
            print("Cannot detect blobs on empty image.")
            return

        modded = detect_blobs(self.image_history[self.curr_index])
        self.notify_mod_operation(modded)

    # Init stuff
    def init_left_buttons(self):
        # Modifiers (row0)
        self.curr_button_row = 0
        self.curr_button_col = 0
        self.add_button(self.left_button_frame, "convolve", self.convolve)
        self.add_button(self.left_button_frame, "gaussian", self.gaussian)
        self.add_button(self.left_button_frame, "med-filter", self.median)
        self.curr_button_col += 2
        self.add_button(self.left_button_frame, "norm_bw", self.norm_hist_bw)
        self.add_button(self.left_button_frame, "norm_col", self.norm_hist_col)

        # Modifiers (row1)
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.left_button_frame, "crop", self.crop)
        self.add_button(self.left_button_frame, "rotate", self.rotate)
        self.add_button(self.left_button_frame, "resize", self.resize)
        self.curr_button_col += 1
        self.add_button(self.left_button_frame, "affine", self.affine_transform)
        self.add_button(self.left_button_frame, "persp_t", self.perspecrive_transform)
        # self.add_button(self.left_button_frame, "affine pixel", self.pixel_transform)
        self.add_button(self.left_button_frame, "draw rect", self.draw_rect)

        # Modifiers (row2)
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.left_button_frame, "inverse_bw", self.inverse_bw)

        # Morphological
        self.add_button(self.left_button_frame, "m_dilate", self.m_dilate)
        self.add_button(self.left_button_frame, "m_erode", self.m_erode)
        self.add_button(self.left_button_frame, "m_open", self.m_open)
        self.add_button(self.left_button_frame, "m_close", self.m_close)
        self.add_button(self.left_button_frame, "m_grad", self.m_grad)
        self.add_button(self.left_button_frame, "m_blackhat", self.m_blackhat)

        # Modifiers (row3)
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.left_button_frame, "to_rgb", self.to_rgb)
        self.add_button(self.left_button_frame, "to_gray", self.to_gray)
        self.add_button(self.left_button_frame, "to_bgr", self.to_bgr)
        self.add_button(self.left_button_frame, "to_hsv", self.to_hsv)
        self.add_button(self.left_button_frame, "to_luv", self.to_luv)
        self.add_button(self.left_button_frame, "to_lab", self.to_lab)
        self.add_button(self.left_button_frame, "to_yuv", self.to_yuv)

    def init_right_buttons(self):
        self.curr_button_row = 0
        self.curr_button_col = 0

        # row1
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.right_button_frame, "corners_st", self.d_corners_st)
        self.add_button(self.right_button_frame, "cornsers_h", self.d_corners_h)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)

        # row2
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.right_button_frame, "edges_manual", self.d_edges_manual)
        self.add_button(self.right_button_frame, "edges_auto", self.d_edges_auto)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)

        # row3
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.right_button_frame, "lines_manual", self.d_lines_manual)
        self.add_button(self.right_button_frame, "lines_auto", self.d_lines_auto)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)

        # row3
        self.curr_button_row += 1
        self.curr_button_col = 0
        self.add_button(self.right_button_frame, "blobs", self.d_blobs)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)
        self.add_button(self.right_button_frame, "placeholder", self.placeholder)

    def placeholder(self):
        print("Congrats, you did nothing.")

    def init_save_laod_buttons(self):
        # Row 1 (save/load)
        self.curr_button_row = 1
        self.curr_button_col = 0
        self.add_button(self.sl_frame0, "save here", lambda: self.save_local(0), "green")
        self.add_button(self.sl_frame0, "load this", lambda: self.load_local(0), "yellow")

        self.curr_button_col = 0
        self.add_button(self.sl_frame1, "save here", lambda: self.save_local(1), "green")
        self.add_button(self.sl_frame1, "load this", lambda: self.load_local(1), "yellow")

        self.curr_button_col = 0
        self.add_button(self.sl_frame2, "save here", lambda: self.save_local(2), "green")
        self.add_button(self.sl_frame2, "load this", lambda: self.load_local(2), "yellow")

        # Row 2 (del)
        self.curr_button_row = 2
        self.curr_button_col = 0
        self.add_button(self.save_frame, "delete save", lambda: self.delete_save(0), "red")
        self.add_button(self.save_frame, "delete save", lambda: self.delete_save(1), "red")
        self.add_button(self.save_frame, "delete save", lambda: self.delete_save(2), "red")

    def init_other_buttons(self):
        self.curr_button_row = 0
        self.curr_button_col = 0
        self.add_button(self.other_button_frame, "Load image", self.load_image, "green", 0)
        self.curr_button_col = 1
        self.add_button(self.other_button_frame, "Save current", self.save_image, "green", 0)

        #spacers
        bg_col = self.other_button_frame["background"]
        self.spacer_frames.append(Frame(self.other_button_frame, width = 10, bg=bg_col))
        self.spacer_frames[-1].grid(row = 0, column = 2)
        self.spacer_frames.append(Frame(self.other_button_frame, width = 10, bg=bg_col))
        self.spacer_frames[-1].grid(row = 0, column = 3)

        # No-op
        self.curr_button_col = 4
        self.add_button(self.other_button_frame, "no-op", self.no_op, "yellow")

        # Reset
        self.curr_button_row = 1
        self.curr_button_col = 0
        self.add_button(self.other_button_frame, "Reset", self.reset, "orange")

        # Clear image
        self.curr_button_row = 1
        self.curr_button_col = 1
        self.add_button(self.other_button_frame, "Clear", self.clear_image, "orange")

        # Undo
        self.curr_button_col = 2
        self.add_button(self.other_button_frame, "Undo", self.undo, "yellow")

        # Redo
        self.curr_button_col = 3
        self.add_button(self.other_button_frame, "Redo", self.redo, "yellow")

        # Spacer
        self.spacer_frames.append(Frame(self.other_button_frame, width = 10, bg=bg_col))
        self.spacer_frames[-1].grid(row = 1, column = 4)

        # Close
        self.curr_button_col = 5
        self.add_button(self.other_button_frame, "Close", self.close_window, "red")


    def add_button(self, frame, text, command, background = "gray", row = None, col = None):
        if row is None:
            row = self.curr_button_row
        if col is None:
            col = self.curr_button_col
            self.curr_button_col += 1

        self.buttons.append(Button(frame, text=text, command=command, bg = background))
        self.buttons[-1].grid(row = row, column = col)


    def save_local(self, index):
        if len(self.image_history) == 0:
            print("Cannot save empty image. ({})".format(index))
            return

        self.saved_image[index] = self.image_history[-1]
        rows, cols = get_image_dimentions(self.saved_image[index])
        self.save_canvas[index].config(width = cols, height = rows)

        self.saved_image_arrays[index] = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.saved_image[index]))
        self.save_canvas[index].delete("all")
        self.save_canvas[index].create_image(0, 0, image=self.saved_image_arrays[index], anchor=tkinter.NW)


    def load_local(self, index):
        if self.saved_image[index] is None:
            print("Cannot load empty image ({})".format(index))
            return

        if not self.has_image():
            self.image_history.append(self.saved_image[index])
            self.draw_origional_image()

        self.notify_mod_operation(self.saved_image[index])

    def delete_save(self, index):
        if self.saved_image[index] is None:
            print("Cannot delete empty save ({})".format(index))
            return

        self.get_choice_from_buttons({"Confirm Clear": True, "Don't clear": False}, callback = lambda: self.do_empty_save(index))

    def do_empty_save(self, index):
        if not self.curr_choice:
            return

        self.save_canvas[index].config(width = 200, height = 200)
        self.saved_image[index] = None
        self.saved_image_arrays[index] = None
        self.save_canvas[index].delete("all")

    def clear_image(self):
        if not self.has_image():
            print("Canvas already clear.")
            return

        self.get_choice_from_buttons({"Confirm Clear": True, "Don't clear": False}, callback = self.do_clear)

    def do_clear(self):
        if not self.curr_choice:
            return

        self.image_history = []
        self.curr_index = 0
        self.origional_image_canvas.delete("all")
        self.origional_image_canvas.config(width = 200, height = 200)
        self.origional_image_label.config(text="Origional (Not to scale)\nOrigional dims: (0x0)")

        self.previous_image_canvas.delete("all")
        self.previous_image_canvas.config(width = 200, height = 200)
        self.previous_image_label.config(text="Previous (0x0)")

        self.current_image_canvas.delete("all")
        self.current_image_canvas.config(width = 200, height = 200)
        self.current_image_label.config(text="Current (0x0)")
