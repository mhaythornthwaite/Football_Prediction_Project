# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:11:31 2020

@author: mhayt
"""

#Access with: http://localhost:5000/

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World! I\'m called Matt'

if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0', port = 5000)
    
    