# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.resalts_table = QtWidgets.QTableView(self.centralwidget)
        self.resalts_table.setGeometry(QtCore.QRect(330, 140, 350, 98))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resalts_table.sizePolicy().hasHeightForWidth())
        self.resalts_table.setSizePolicy(sizePolicy)
        self.resalts_table.setObjectName("resalts_table")
        self.label_predicted = QtWidgets.QLabel(self.centralwidget)
        self.label_predicted.setGeometry(QtCore.QRect(20, 40, 95, 14))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_predicted.sizePolicy().hasHeightForWidth())
        self.label_predicted.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_predicted.setFont(font)
        self.label_predicted.setObjectName("label_predicted")
        self.label_expert = QtWidgets.QLabel(self.centralwidget)
        self.label_expert.setGeometry(QtCore.QRect(20, 70, 95, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_expert.setFont(font)
        self.label_expert.setObjectName("label_expert")
        self.path_predicted_line = QtWidgets.QLineEdit(self.centralwidget)
        self.path_predicted_line.setGeometry(QtCore.QRect(140, 40, 91, 22))
        self.path_predicted_line.setObjectName("path_predicted_line")
        self.path_expert_line = QtWidgets.QLineEdit(self.centralwidget)
        self.path_expert_line.setGeometry(QtCore.QRect(140, 70, 91, 22))
        self.path_expert_line.setObjectName("path_expert_line")
        self.explore_path_predicted = QtWidgets.QPushButton(self.centralwidget)
        self.explore_path_predicted.setGeometry(QtCore.QRect(260, 40, 101, 22))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.explore_path_predicted.sizePolicy().hasHeightForWidth())
        self.explore_path_predicted.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.explore_path_predicted.setFont(font)
        self.explore_path_predicted.setObjectName("explore_path_predicted")
        self.explore_path_expert = QtWidgets.QPushButton(self.centralwidget)
        self.explore_path_expert.setGeometry(QtCore.QRect(260, 70, 101, 22))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.explore_path_expert.sizePolicy().hasHeightForWidth())
        self.explore_path_expert.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.explore_path_expert.setFont(font)
        self.explore_path_expert.setObjectName("explore_path_expert")
        self.evaluate_button = QtWidgets.QPushButton(self.centralwidget)
        self.evaluate_button.setGeometry(QtCore.QRect(270, 100, 101, 22))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.evaluate_button.sizePolicy().hasHeightForWidth())
        self.evaluate_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.evaluate_button.setFont(font)
        self.evaluate_button.setObjectName("evaluate_button")
        self.save_to_xls = QtWidgets.QPushButton(self.centralwidget)
        self.save_to_xls.setGeometry(QtCore.QRect(270, 130, 80, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.save_to_xls.setFont(font)
        self.save_to_xls.setObjectName("save_to_xls")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_predicted.setText(_translate("MainWindow", "Оцениваемая разметка"))
        self.label_expert.setText(_translate("MainWindow", "Разметка эксперта"))
        self.explore_path_predicted.setText(_translate("MainWindow", "ОБЗОР"))
        self.explore_path_expert.setText(_translate("MainWindow", "ОБЗОР"))
        self.evaluate_button.setText(_translate("MainWindow", "ОЦЕНИТЬ РАЗМЕТКУ"))
        self.save_to_xls.setText(_translate("MainWindow", "СОХРАНИТЬ"))
