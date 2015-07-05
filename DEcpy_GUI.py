from __future__ import division
from Tkinter import *
import numpy as np
import pylab as py
import os
import sys
'''
class DEcpy_UI:
    def __init__(self, parent):
        """Initiate the instance of Tkinter"""
        Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()
    
    def __init__(self, master):
        frame = Frame(master, height = 475, width = 700)
        frame.pack()
        bottomframe = Frame(root)
        redbutton = Button(frame, text="Red", fg="red")
        redbutton.pack( side = LEFT)

        greenbutton = Button(frame, text="Brown", fg="brown")
        greenbutton.pack( side = LEFT )

        bluebutton = Button(frame, text="Blue", fg="blue")
        bluebutton.pack( side = LEFT )

        blackbutton = Button(bottomframe, text="Black", fg="black")
        blackbutton.pack( side = BOTTOM)
        
        hi_there = Button(frame, text="Hello", command=self.say_hi)
        hi_there.pack(side=LEFT)

    def say_hi(self):
        print "hi there, everyone!"


if __name__ == "__main__":
    root = Tk()
    app = DEcpy_UI(root)
    root.mainloop()
'''
class DEcpy_UI(Frame):
    
    def __init__(self, parent):
        Frame.__init__(self, parent, background="gray")
        
        self.parent = parent
        self.parent.title("Dark Energy Calculator")
        self.pack(fill=BOTH, expand=1)
        self.centerWindow()
    
    def centerWindow(self):
        self.grid()
        self.Home = Frame(self, height = 500, width = 800)

        self.Home.grid(column = 0, row = 0, columnspan = 3, sticky = "N")
        self.Home.grid_propagate(0)
        '''
        w = 800
        h = 500
        
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        
        x = (sw - w)/2
        y = (sh - h)/2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))
        '''
        self.IntroStr1 = StringVar()
        self.Welcome1 = Label(self.Home, textvariable = self.IntroStr1, anchor = "center", fg = "black")
        self.Welcome1.grid(column = 0, row = 0, columnspan = 4)
        self.IntroStr1.set(u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        
        self.IntroStr2 = StringVar()
        self.Welcome2 = Label(self.Home, textvariable = self.IntroStr2, anchor = "center", fg = "black")
        self.Welcome2.grid(column = 0, row = 1, columnspan = 4, pady = 3)
        self.IntroStr2.set(u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                                            
        self.IntroStr3 = StringVar()
        self.Welcome3 = Label(self.Home, textvariable = self.IntroStr3, anchor = "center", fg = "black")
        self.Welcome3.grid(column = 0, row = 2, columnspan = 4, pady = 3)
        self.IntroStr3.set(u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        
        # construct the button to choose a model
        self.SinglePage = Button(self.Home, text = u"Run a Single Model", command = self.Single1, width = 35)
        self.SinglePage.grid(column = 2, row = 3, padx = 0, pady = 0)
        
        # create the window for the run
        self.SingleInput = Frame(self, height = 475, width = 700)
        self.SingleInput.grid(column = 0, row = 0, columnspan = 3, sticky = "N")
        self.SingleInput.grid_propagate(0)
            
            
            
        self.statusVariable = StringVar()
        self.status = Label(self, textvariable = self.statusVariable, anchor = "w", fg = "yellow", bg = "blue")
        self.status.grid(column = 0, row = 10,columnspan = 5, sticky = 'EWS')
        self.statusVariable.set(u"Welcome to DE_Mod_Gen")
            


    def Single1(self):
        """Single1 takes the user to the Single Model window"""
        self.statusVariable.set("Run a single model with your chosen parameters")
        self.Home.grid_remove()
        self.Result.grid_remove()
        self.SingleInput.grid()
        self.Singlex1Ent.focus_set()
        self.Singlex1Ent.selection_range(0, END)


if __name__ == '__main__':
    root = Tk()
    ex = DEcpy_UI(root)
    root.mainloop()

