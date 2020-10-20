# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

# ---- Third party imports
from PyQt5.QtWidgets import (
    QGridLayout, QWidget, QPushButton, QStyle, QScrollArea)


class ToolPanel(QWidget):
    """
    A custom widget that mimicks the behavior of the "Tools" sidepanel in
    Adobe Acrobat. It is derived from a QToolBox with the following variants:

    1. Only one tool can be displayed at a time.
    2. Unlike the stock QToolBox widget, it is possible to hide all the tools.
    3. It is also possible to hide the current displayed tool by clicking on
       its header.
    4. The tools that are hidden are marked by a right-arrow icon, while the
       tool that is currently displayed is marked with a down-arrow icon.
    5. Closed and Expanded arrows can be set from custom icons.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.__iclosed = QWidget().style().standardIcon(
            QStyle.SP_ToolBarHorizontalExtensionButton)
        self.__iexpand = QWidget().style().standardIcon(
            QStyle.SP_ToolBarVerticalExtensionButton)

        self.setLayout(QGridLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)  # (l, t, r, b)

        self.__currentIndex = -1

    def setIcons(self, ar_right, ar_down):
        self.__iclosed = ar_right
        self.__iexpand = ar_down

    def addItem(self, tool, text):

        N = self.layout().rowCount()

        # Add Header :

        head = QPushButton(text)
        head.setIcon(self.__iclosed)
        head.clicked.connect(self.__isClicked__)
        head.setStyleSheet("QPushButton {text-align:left;}")

        self.layout().addWidget(head, N-1, 0)

        # Add Item in a ScrollArea :

        scrollarea = QScrollArea()
        scrollarea.setFrameStyle(0)
        scrollarea.hide()
        scrollarea.setStyleSheet("QScrollArea {background-color:transparent;}")
        scrollarea.setWidgetResizable(True)

        tool.setObjectName("myViewport")
        tool.setStyleSheet("#myViewport {background-color:transparent;}")
        scrollarea.setWidget(tool)

        self.layout().addWidget(scrollarea, N, 0)
        self.layout().setRowStretch(N+1, 100)

    def __isClicked__(self):

        for row in range(0, self.layout().rowCount()-1, 2):

            head = self.layout().itemAtPosition(row, 0).widget()
            tool = self.layout().itemAtPosition(row+1, 0).widget()

            if head == self.sender():
                if self.__currentIndex == row:
                    # if clicked tool is open, close it
                    head.setIcon(self.__iclosed)
                    tool.hide()
                    self.__currentIndex = -1
                else:
                    # if clicked tool is closed, expand it
                    head.setIcon(self.__iexpand)
                    tool.show()
                    self.__currentIndex = row
            else:
                # close all the other tools so that only one tool can be
                # expanded at a time.
                head.setIcon(self.__iclosed)
                tool.hide()
