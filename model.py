import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from keras.models import load_model 

class Algo:
    def __init__(self,name):
        self.name = name
        

    def cnn_algo(self):
        # Recreate the exact same model, including its weights and the optimizer
        
        classes =['Adhirasam', 'Aloo_Gobi', 'Aloo_Matar', 'Aloo_Methi', 'Aloo_Shimla_Mirch', 
                  'Aloo_Tikki', 'Butter_Chicken', 'Chapati', 'Cheesecake', 'Chicken_Biryani', 
                  'Chicken_Curry', 'Chicken_Tikka', 'Chicken_Tikka_Masala', 'Chicken_Wings', 'Chocolate_Cake', 
                  'Dal_Tadka', 'Dharwad_Pedha', 'Donuts', 'Dumplings', 'Egg_Fried_Rice', 
                  'French_Fries', 'Grilled_Cheese_Sandwich', 'Hamburger', 'Ice_Cream', 'Kachori', 
                  'Lassi', 'Mysore_Pak', 'Naan', 'Navrattan_Korma', 'Omelette', 
                  'Palak_Paneer', 'Pancakes', 'Paneer_Butter_Masala', 'Pizza', 'Ramen', 
                  'Ras_Malai', 'Rasgulla', 'Samosa', 'Sushi']


        uses={'Adhirasam': 'Adhirasam - per pic 1 Piece (approx. 45g) = Calories (kcal) = 200 kcal, Carbohydrates = 28.0g ,Protein = 2.0g ,Sugar = 18.0g ,Fats = 8.0g.',

        'Aloo_Gobi':'Aloo Gobi - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 130 kcal, Carbohydrates = 14.0g, Protein = 3.0g, Sugar = 2.5g, Fats = 7.0g.',

        'Aloo_Matar':'Aloo Matar - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 130 kcal, Carbohydrates = 16.0g, Protein = 3.5g, Sugar = 3.0g, Fats = 6.0g',

        'Aloo_Methi':'Aloo Methi - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 115 kcal, Carbohydrates = 12.0g, Protein = 3.0g, Sugar = 1.0g, Fats = 6.0g.',
        
        'Aloo_Shimla_Mirch':'Aloo Shimla Mirch - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 115 kcal, Carbohydrates = 13.0g, Protein = 2.5g, Sugar = 3.0g, Fats = 6.0g.',

        'Aloo_Tikki':'Aloo Tikki - 1 Standard Piece (approx. 60g) = Calories (kcal) = 135 kcal, Carbohydrates = 18.0g, Protein = 2.0g, Sugar =0.5g, Fats =6.0g.',

        'Butter_Chicken':'Butter Chicken - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 260 kcal, Carbohydrates = 9.0g, Protein = 15.0g, Sugar = 5.0g, Fats = 18.0g.',

        'Chapati':'Chapati - Per Pic (approx. 40g) = Calories (kcal) = 90 kcal, Carbohydrates = 18.0g, Protein = 3.0g, Sugar = 0.2g, Fats = 0.5g.',

        'Cheesecake':'Cheesecake - per pic 1 Slice (approx. 100g) = Calories (kcal) = 350 kcal, Carbohydrates = 32.0g, Protein = 6.0g, Sugar = 21.0g, Fats = 22.0g.',

        'Chicken_Biryani':'Chicken Biryani - 1 Standard Plate (approx. 200g) = Calories (kcal) = 290 kcal, Carbohydrates = 36.0g, Protein = 14.0g, Sugar = 1.5g, Fats = 10.0g.',

        'Chicken_Curry':'Chicken Curry - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 185 kcal, Carbohydrates = 6.0g, Protein = 18.0g, Sugar = 2.0g, Fats = 10.0g.',

        'Chicken_Tikka':'Chicken Tikka - 1 Piece / Skewer (approx. 50g) = Calories (kcal) = 85 kcal, Carbohydrates = 2.0g, Protein = 11.0g, Sugar = 0.5g, Fats = 3.5g.',

        'Chicken_Tikka_Masala':'Chicken Tikka Masala - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 220 kcal, Carbohydrates = 8.0g, Protein = 16.0g, Sugar = 4.0g, Fats = 14.0g.',

        'Chicken_Wings':'Chicken Wings- 1 Piece Fried & Sauced (35g) = Calories (kcal) = 100 kcal, Carbohydrates = 3.0g, Protein = 6.0g, Sugar = 1.0g, Fats = 7.0g.',

        'Chocolate_Cake':'Chocolate Cake - per pic 1 Slice (approx. 65g) = Calories (kcal) = 250 kcal, Carbohydrates = 35.0g, Protein = 3.0g, Sugar = 22.0g, Fats = 11.0g.',

        'Dal_Tadka':'Dal Tadka - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 150 kcal, Carbohydrates = 18.0g, Protein = 7.0g, Sugar = 1.0g, Fats = 6.0g.',

        'Dharwad_Pedha':'Dharwad Peda - Per Pic (approx. 25g) = Calories (kcal) = 115 kcal, Carbohydrates = 15.0g, Protein = 3.0g, Sugar = 15.0g, Fats = 4.6g.',

        'Donuts':'Donuts - Per pic 1 Glazed (approx. 60g) = Calories (kcal) = 240 kcal, Carbohydrates = 30.0g, Protein = 3.0g, Sugar = 15.0g, Fats = 12.0g.',

        'Dumplings':'Dumpling - Per Pic = Calories (kcal) = 40 kcal, Carbohydrates = 4.5g, Protein = 4.5g, Sugar = 2.0g, Fats = 0.3g.',

        'Egg_Fried_Rice':'EGG Fried Rice(85g) - 1 Small Bowl (approx. 150g) = Calories (kcal) = 230 kcal, Carbohydrates = 38.0g, Protein = 4.0g, Sugar = 1.0g, Fats = 7.0g.',

        'French_Fries':'French Fries - 1 Medium Order (approx. 85g) = Calories (kcal) = 210 kcal, Carbohydrates = 26.0g, Protein = 2.5g, Sugar = 0.3g, Fats = 11.0g.',

        'Grilled_Cheese_Sandwich':'Grilled Cheese - per pic 1 Sandwich (approx. 110g) = Calories (kcal) = 350 kcal, Carbohydrates = 30.0g, Protein = 12.0g, Sugar = 3.0g ,Fats = 18.0g.',

        'Hamburger':'Hamburger - per pic 1 Standard Beef (approx. 100g) = Calories (kcal) = 290 kcal, Carbohydrates = 30.0g, Protein = 15.0g, Sugar = 5.0g, Fats = 12.0g.',

        'Ice_Cream':'Ice Cream - per pic 1 Scoop Vanilla (approx. 70g) = Calories (kcal)= 140 kcal ,Carbohydrates = 16.0g, Protein = 2.5g, Sugar = 14.0g, Fats = 7.0g.',

        'Kachori':'Kachori - per pic 1 Piece (approx. 75g) = Calories (kcal) = 300 kcal, Carbohydrates = 35.0g, Protein = 5.0g, Sugar = 1.0g, Fats = 15.0g.',

        'Lassi':'Lassi (Sweet)(approx. 250ml) = Calories (kcal) = 200 kcal, Carbohydrates = 32.0g, Protein = 6.0g, Sugar = 28.0g, Fats = 5.0g.',

        'Mysore_Pak':'Mysore Pak - Per Pic 1 Piece (approx. 30g) (30g), Calories (kcal) = 140 kcal, Carbohydrates = 17.1g, Protein = 1.5g, Sugar = 14.0g, Fats = 7.1g.',

        'Naan':'Naan - Per Pic 1 Piece (approx. 90g) = Calories (kcal) = 260 kcal, Carbohydrates = 45.0g, Protein = 7.5g, Sugar = 2.5g, Fats = 4.0g.',

        'Navrattan_Korma':'Navrattan Korma - 1Serving Bowl (approx. 150g) = Calories (kcal) = 180 kcal, Carbohydrates = 14.0g, Protein = 4.0g, Sugar = 5.0g, Fats = 12.0g.',

        'Omelette':'Omelet - per pic 1 Plain 2-Egg (approx. 100g) = Calories (kcal) = 150 kcal, Carbohydrates = 1.0g, Protein = 12.0g, Sugar = 0.5g, Fats = 11.0g.',

        'Palak_Paneer':'Palak Paneer - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 200 kcal, Carbohydrates = 6.0g, Protein = 9.0g, Sugar = 1.5g, Fats = .0g.',

        'Pancakes':'Pancake - 1 Standard 6" (approx. 45g) = Calories (kcal) = 125 kcal, Carbohydrates = 22.0g, Protein = 3.0g, Sugar = 5.0g, Fats = 3.0g.',

        'Paneer_Butter_Masala':'Paneer Butter Masala - 1 Serving Bowl (approx. 150g) = Calories (kcal) = 280 kcal, Carbohydrates = 10.0g, Protein = 11.0g, Sugar = 4.0g, Fats = 22.0g.',

        'Pizza':'Pizza - Per Slice = Calories (kcal) = 270 kcal, Carbohydrates = 32.0g, Protein = 12.0g, Sugar = 3.5g, Fats = 10.0g.',

        'Ramen':'Ramen - 1 Loaded Bowl (approx. 350g) = Calories (kcal) = 350 kcal, Carbohydrates = 45.0g, Protein = 14.0g, Sugar = 2.0g, Fats = 12.0g.',

        'Ras_Malai':'Ras Malai - 1 Piece + Syrup (approx. 75g) = Calories (kcal) = 140 kcal, Carbohydrates = 18.0g, Protein = 4.0g, Sugar = 15.0g, Fats = 6.0g.',

        'Rasgulla':'Rasgulla - per pic 1 Piece (approx. 50g) = Calories (kcal) = 80 kcal, Carbohydrates = 15.0g, Protein = 2.0g, Sugar = 14.0g, Fats = 1.0g.',

        'Samosa':'Samosa - 1 Standard Piece (approx. 75g) = Calories (kcal) = 220 kcal, Carbohydrates = 24.0g, Protein = 3.5g, Sugar = 1.0g, Fats = 12.0g.',

        'Sushi':'Sushi- 1 Piece California Roll (30g) = Calories (kcal) = 40 kcal, Carbohydrates = 6.0g, Protein = 1.0g, Sugar = 1.0g, Fats = 1.0g.'}

        da = self.name
        print(da)
        directory = os.getcwd()
        print(directory)

        #path = 'Sign_Modified/BA/BA1-removebg-preview (1).png'
        loaded_model = load_model("Model/food_0250.h5")
        path = 'upload/'+self.name
        print(path)

        img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)


        # new_model = tf.keras.models.load_model('my_model.h5')
        
        predictions = loaded_model.predict(img_array)

        print(predictions[0]*100, "\n", classes)
        print("Prediction: ", classes[np.argmax(predictions)], f"{predictions[0][np.argmax(predictions)]*100}%")
        res= classes[np.argmax(predictions)]
        # uses=res[0]
        print(res)
        return res,uses[res]
       
        