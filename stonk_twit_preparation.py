import pickle
import pandas as pd
from text_mining import TextCleaner
from datetime import datetime


class StonkTwit(TextCleaner):

    def __init__(self):
        with open("cols_model.pkl", "rb") as c:
            self.__cols = pickle.load(c)

        with open("stonk_rf.pkl", "rb") as s:
            self.__model = pickle.load(s)

    def predict_stonk_probability(self, text, with_media=False):
        input_text_list = list(self.get_clean_text(text=text).word)
        input_df = pd.DataFrame(columns=self.__cols, data=[[0] * len(self.__cols)])
        for word in set(input_text_list):
            for c in self.__cols:
                if word.strip() == c:
                    input_df[c] = 1
        today = datetime.now()
        input_df[f"hour {today.hour}"] = 1
        input_df[f"day_name {today.strftime('%A')}"] = 1
        if with_media:
            input_df['is_media'] = 1
        stonk_proba = self.__model.predict_proba(input_df)[:, 1]
        return round(stonk_proba[0], 3)