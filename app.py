#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:16:01 2023

@author: nehachavan
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

loaded_model = pickle.load(open("C:\Users\shubh\Downloads\Employee_Attrition\last123.sav", 'rb'))


def EMPLOYEE_predict(input_data):
    #input_data=(0.22,0.576,0.)
    # changing input data into numpy array
    np_array = np.asarray(input_data)
    # reshape array as we are predecting for one instance
    input_reshaped = np_array.reshape(1, -1)
    predict = loaded_model.predict(input_reshaped)
    return predict


def main():
    # giving title
    st.title("EMPLOYEE PREDECTION WEB APP")
    Age = st.text_input("AGE")
    DistanceFromHome = st.text_input('Distance From Home')
    MonthlyIncome = st.text_input("Monthly Income")
    MonthlyRate = st.text_input('Monthly Rate')
    WorkLifeBalance = st.text_input('Work Life Balance')


    empty = ''

    if st.button("EMPLOYEE predection result"):
        empty = EMPLOYEE_predict([Age,  DistanceFromHome,
                                     MonthlyIncome, MonthlyRate,
                                      WorkLifeBalance])

    st.success(empty)


if __name__ == '__main__':
    main()