import numpy as np
from PIL import Image 
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as implt
import warnings
warnings.filterwarnings('ignore')



def testing():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename()
    
    model = tf.keras.models.load_model("bitirme4.model")
    
    # plot 
    covid1=implt.imread(file_path)
    plt.subplot(1,2,1)
    plt.title('Selected CT')
    plt.imshow(covid1)
    #plt.show()
    print(covid1.shape)

    test=[]
    test1=np.ones(1)
    img_size = 64

    covid=Image.open(file_path).convert('L') # converting grey scale
    covid=covid.resize((img_size,img_size),Image.ANTIALIAS) # resizing to 50,50
    covid=np.asarray(covid)/255 # bit format
    test.append(covid)
    
    X_test1=np.concatenate((test),axis=0)
    #Y_test1=np.concatenate((test),axis=0).reshape(X_test1.shape[0],1)

    print('X test shape:',X_test1.shape)
    #print('Y test shape:',Y_test1.shape)
    
    X_test1 = X_test1.reshape(-1,64,64,1)
    print("x_test shape: ",X_test1.shape)

    test_sonuclari = model.predict(X_test1)
    
    if (test_sonuclari[0,0]) > (test_sonuclari[0,1]):
        print("Test sonucu negatiftir.")  
    else:
        print("Test sonucu pozitiftir.")
    
    afunc()
        
def afunc():
    a = input("press 'y' if you want to use and 'n' if you don't\n")
    
    if(a=="y"):
        testing()
    elif(a=="n"):
        print("have a nice day")  

  
afunc()















