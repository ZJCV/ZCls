# -*- coding: utf-8 -*-

"""
@date: 2020/10/22 上午9:24
@file: img_visualizer.py
@author: zj
@description: 
"""

import itertools
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import torch
from detectron2.utils.visualizer import Visualizer


class ImgVisualizer(Visualizer):
    def __init__(self, img_rgb, meta, **kwargs):
        """
        See https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/visualizer.py
        for more details.
        Args:
            img_rgb: a tensor or numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            meta (MetadataCatalog): image metadata.
                See https://github.com/facebookresearch/detectron2/blob/81d5a87763bfc71a492b5be89b74179bd7492f6b/detectron2/data/catalog.py#L90
        """
        super(ImgVisualizer, self).__init__(img_rgb, meta, **kwargs)

    def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="w",
            horizontal_alignment="center",
            vertical_alignment="bottom",
            box_facecolor="black",
            alpha=0.5,
    ):
        """
        Draw text at the specified position.
        Args:
            text (str): the text to draw on image.
            position (list of 2 ints): the x,y coordinate to place the text.
            font_size (Optional[int]): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color (str): color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`.
            vertical_alignment (str): see `matplotlib.text.Text`.
            box_facecolor (str): color of the box wrapped around the text. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
            alpha (float): transparency level of the box.
        """
        if not font_size:
            font_size = self._default_font_size
        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="monospace",
            bbox={
                "facecolor": box_facecolor,
                "alpha": alpha,
                "pad": 0.7,
                "edgecolor": "none",
            },
            verticalalignment=vertical_alignment,
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
        )

    def draw_multiple_text(
            self,
            text_ls,
            box_coordinate,
            *,
            top_corner=True,
            font_size=None,
            color="w",
            box_facecolors="black",
            alpha=0.5,
    ):
        """
        Draw a list of text labels for some bounding box on the image.
        Args:
            text_ls (list of strings): a list of text labels.
            box_coordinate (tensor): shape (4,). The (x_left, y_top, x_right, y_bottom)
                coordinates of the box.
            top_corner (bool): If True, draw the text labels at (x_left, y_top) of the box.
                Else, draw labels at (x_left, y_bottom).
            font_size (Optional[int]): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color (str): color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            box_facecolors (str): colors of the box wrapped around the text. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
            alpha (float): transparency level of the box.
        """
        if not isinstance(box_facecolors, list):
            box_facecolors = [box_facecolors] * len(text_ls)
        assert len(box_facecolors) == len(
            text_ls
        ), "Number of colors provided is not equal to the number of text labels."
        if not font_size:
            font_size = self._default_font_size
        text_box_width = font_size + font_size // 2
        # If the texts does not fit in the assigned location,
        # we split the text and draw it in another place.
        if top_corner:
            num_text_split = self._align_y_top(
                box_coordinate, len(text_ls), text_box_width
            )
            y_corner = 1
        else:
            num_text_split = len(text_ls) - self._align_y_bottom(
                box_coordinate, len(text_ls), text_box_width
            )
            y_corner = 3

        text_color_sorted = sorted(
            zip(text_ls, box_facecolors), key=lambda x: x[0], reverse=True
        )
        if len(text_color_sorted) != 0:
            text_ls, box_facecolors = zip(*text_color_sorted)
        else:
            text_ls, box_facecolors = [], []
        text_ls, box_facecolors = list(text_ls), list(box_facecolors)
        self.draw_multiple_text_upward(
            text_ls[:num_text_split][::-1],
            box_coordinate,
            y_corner=y_corner,
            font_size=font_size,
            color=color,
            box_facecolors=box_facecolors[:num_text_split][::-1],
            alpha=alpha,
        )
        self.draw_multiple_text_downward(
            text_ls[num_text_split:],
            box_coordinate,
            y_corner=y_corner,
            font_size=font_size,
            color=color,
            box_facecolors=box_facecolors[num_text_split:],
            alpha=alpha,
        )

    def draw_multiple_text_upward(
            self,
            text_ls,
            box_coordinate,
            *,
            y_corner=1,
            font_size=None,
            color="w",
            box_facecolors="black",
            alpha=0.5,
    ):
        """
        Draw a list of text labels for some bounding box on the image in upward direction.
        The next text label will be on top of the previous one.
        Args:
            text_ls (list of strings): a list of text labels.
            box_coordinate (tensor): shape (4,). The (x_left, y_top, x_right, y_bottom)
                coordinates of the box.
            y_corner (int): Value of either 1 or 3. Indicate the index of the y-coordinate of
                the box to draw labels around.
            font_size (Optional[int]): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color (str): color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            box_facecolors (str or list of strs): colors of the box wrapped around the text. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
            alpha (float): transparency level of the box.
        """
        if not isinstance(box_facecolors, list):
            box_facecolors = [box_facecolors] * len(text_ls)
        assert len(box_facecolors) == len(
            text_ls
        ), "Number of colors provided is not equal to the number of text labels."

        assert y_corner in [1, 3], "Y_corner must be either 1 or 3"
        if not font_size:
            font_size = self._default_font_size

        x, horizontal_alignment = self._align_x_coordinate(box_coordinate)
        y = box_coordinate[y_corner].item()
        for i, text in enumerate(text_ls):
            self.draw_text(
                text,
                (x, y),
                font_size=font_size,
                color=color,
                horizontal_alignment=horizontal_alignment,
                vertical_alignment="bottom",
                box_facecolor=box_facecolors[i],
                alpha=alpha,
            )
            y -= font_size + font_size // 2

    def draw_multiple_text_downward(
            self,
            text_ls,
            box_coordinate,
            *,
            y_corner=1,
            font_size=None,
            color="w",
            box_facecolors="black",
            alpha=0.5,
    ):
        """
        Draw a list of text labels for some bounding box on the image in downward direction.
        The next text label will be below the previous one.
        Args:
            text_ls (list of strings): a list of text labels.
            box_coordinate (tensor): shape (4,). The (x_left, y_top, x_right, y_bottom)
                coordinates of the box.
            y_corner (int): Value of either 1 or 3. Indicate the index of the y-coordinate of
                the box to draw labels around.
            font_size (Optional[int]): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color (str): color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            box_facecolors (str): colors of the box wrapped around the text. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
            alpha (float): transparency level of the box.
        """
        if not isinstance(box_facecolors, list):
            box_facecolors = [box_facecolors] * len(text_ls)
        assert len(box_facecolors) == len(
            text_ls
        ), "Number of colors provided is not equal to the number of text labels."

        assert y_corner in [1, 3], "Y_corner must be either 1 or 3"
        if not font_size:
            font_size = self._default_font_size

        x, horizontal_alignment = self._align_x_coordinate(box_coordinate)
        y = box_coordinate[y_corner].item()
        for i, text in enumerate(text_ls):
            self.draw_text(
                text,
                (x, y),
                font_size=font_size,
                color=color,
                horizontal_alignment=horizontal_alignment,
                vertical_alignment="top",
                box_facecolor=box_facecolors[i],
                alpha=alpha,
            )
            y += font_size + font_size // 2

    def _align_x_coordinate(self, box_coordinate):
        """
            Choose an x-coordinate from the box to make sure the text label
            does not go out of frames. By default, the left x-coordinate is
            chosen and text is aligned left. If the box is too close to the
            right side of the image, then the right x-coordinate is chosen
            instead and the text is aligned right.
            Args:
                box_coordinate (array-like): shape (4,). The (x_left, y_top, x_right, y_bottom)
                coordinates of the box.
            Returns:
                x_coordinate (float): the chosen x-coordinate.
                alignment (str): whether to align left or right.
        """
        # If the x-coordinate is greater than 5/6 of the image width,
        # then we align test to the right of the box. This is
        # chosen by heuristics.
        if box_coordinate[0] > (self.output.width * 5) // 6:
            return box_coordinate[2], "right"

        return box_coordinate[0], "left"

    def _align_y_top(self, box_coordinate, num_text, textbox_width):
        """
            Calculate the number of text labels to plot on top of the box
            without going out of frames.
            Args:
                box_coordinate (array-like): shape (4,). The (x_left, y_top, x_right, y_bottom)
                coordinates of the box.
                num_text (int): the number of text labels to plot.
                textbox_width (float): the width of the box wrapped around text label.
        """
        dist_to_top = box_coordinate[1]
        num_text_top = dist_to_top // textbox_width

        if isinstance(num_text_top, torch.Tensor):
            num_text_top = int(num_text_top.item())

        return min(num_text, num_text_top)

    def _align_y_bottom(self, box_coordinate, num_text, textbox_width):
        """
            Calculate the number of text labels to plot at the bottom of the box
            without going out of frames.
            Args:
                box_coordinate (array-like): shape (4,). The (x_left, y_top, x_right, y_bottom)
                coordinates of the box.
                num_text (int): the number of text labels to plot.
                textbox_width (float): the width of the box wrapped around text label.
        """
        dist_to_bottom = self.output.height - box_coordinate[3]
        num_text_bottom = dist_to_bottom // textbox_width

        if isinstance(num_text_bottom, torch.Tensor):
            num_text_bottom = int(num_text_bottom.item())

        return min(num_text, num_text_bottom)
