from keras import backend as K

def R2(y_true, y_pred):
    ''' R^2 (coefficient of determination) regression score function. To avoid NaN, added 1e-8 to denominator ''' 
    SS_num =  K.sum(K.square(y_true-y_pred)) 
    SS_den = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_num/(SS_den + K.epsilon())
