'''
Created on 25-Sep-2013

@author: panache
'''

import math 
from itertools import repeat
import matplotlib.pyplot as plt
#***************************************************************************************************************************** 
    #Helper Functions
#*****************************************************************************************************************************   

#===============================================================================
# readKFolds
# This function reads the emails from spam.data file into 10 folds.
#===============================================================================
def readKFolds():
    emaildata = open('C:/Users/panache/Desktop/CourseWork/Machine Learning/HW2-Spam Filter/spambase.data','r')
    count = 0
    #creating a list of list with 10 empty lists
    folds = [[] for i in repeat(None, 10)]
    for line in emaildata:
        folds[count].append(readEmail(line))
        if count == 9:
            count = 0
        else:
            count = count +1
    return folds

#This function processes each line from file to create a list of feature values
#===============================================================================
# readEmail
#===============================================================================
def readEmail(line):
    email = line.strip()
    feature_values = email.split(',')
    return feature_values

#===============================================================================
# getActualResult
#===============================================================================
def getActualResult(testing_fold):
    act_result = []
    for email in testing_fold:
        act_result.append(email[57])
    return act_result


#===============================================================================
# calculateErrorRate
#===============================================================================
def calculateErrorRate(cal_result,act_result):
    act_spamcount =0
    act_hamcount = 0
    cal_spamcount=0
    cal_hamcount = 0
    i=0
    while i< len(cal_result):
        if int(act_result[i]) == 1:          
            act_spamcount += 1
        if cal_result[i] == 1:
            cal_spamcount += 1
        i += 1
    
    act_hamcount = len(act_result) - act_spamcount   
    cal_hamcount = len(cal_result) - cal_spamcount
    #print("Act. Spamcount = %s v/s Cal. Spamcount = %s" %(act_spamcount,cal_spamcount))
    #print("Act. hamcount = %s v/s Cal. hamcount = %s" %(act_hamcount,cal_hamcount))
    truePositive = 0.0
    falsePositive = 0.0
    trueNegative = 0.0
    falseNegative = 0.0
    for i in range(0,len(cal_result)):
        if (cal_result[i] == 1) and (int(act_result[i])== 1):
            truePositive += 1
        elif (cal_result[i] == 1) and (int(act_result[i]) == 0):
            falsePositive += 1
        elif (cal_result[i] == 0) and (int(act_result[i]) == 1):
            falseNegative += 1
        elif (cal_result[i] == 0) and (int(act_result[i]) == 0):
            trueNegative += 1
    errorRate = float(falseNegative+falsePositive)/float(falseNegative+falsePositive+truePositive+trueNegative)
    
    return errorRate,falsePositive, falseNegative,falsePositive/(falsePositive+trueNegative),falseNegative/(falseNegative+truePositive) 


#===============================================================================
# genrateROC
#===============================================================================
def genrateROC(cal_result_values,act_result):
    thresholds = sorted(cal_result_values, reverse = True)
    i =0
    val =0
    #===========================================================================
    # filex = open('roc-points-x.txt', mode='a')
    # filey = open('roc-points-y.txt', mode='a')
    #===========================================================================
    roc_points = []
    for val in range(0, len(cal_result_values)):
        predicted_values = []
        for i in range(0, len(cal_result_values)):
            if cal_result_values[i] >= thresholds[val]:
                predicted_values.append(1)
            else:
                predicted_values.append(0)
        truePositive = 0.0
        falsePositive = 0.0
        trueNegative = 0.0
        falseNegative = 0.0
        for i in range(0,len(predicted_values)):
            if (predicted_values[i] == 1) and (int(act_result[i])== 1):
                truePositive += 1
            elif (predicted_values[i] == 1) and (int(act_result[i]) == 0):
                falsePositive += 1
            elif (predicted_values[i] == 0) and (int(act_result[i]) == 1):
                falseNegative += 1
            elif (predicted_values[i] == 0) and (int(act_result[i]) == 0):
                trueNegative += 1
        roc_points.append((float(falsePositive)/(trueNegative + falsePositive),float(truePositive)/(truePositive + falseNegative)))
        #plot(roc_points)
    xlist =[]
    ylist =[]
    for point in roc_points:
        xlist.append(float(point[0]))
        ylist.append(float(point[1]))
        
         #======================================================================
         # filex.write(str(point[0])+'\n')
         # filey.write(str(point[1])+'\n')
         #======================================================================
    plt.xlabel('FPR (1-Specificity)')
    plt.ylabel('TPR (Sensitivity)')
    plt.title('ROC for Compairing 3 Classifiers')
    plt.grid(True)
    plt.plot(xlist,ylist)
    findAUC(roc_points)
    #plt.show()
    #===========================================================================
    # filex.close() 
    # filex.close()    
    #===========================================================================
    return roc_points   
#def plot(roc_points):
    
    
    
def findAUC(roc_points):
    area = 0.0
    for i in range(1,len(roc_points)):
        area += (roc_points[i][0]-roc_points[i-1][0])*(roc_points[i][1]+roc_points[i-1][1])
    area = 0.5*area
    print("Area under the ROC = " ,area)
            
                
    
    
    
    
    
  
#===============================================================================
# calculateProbablities
#===============================================================================
def calculateProbablities (excludedfold,folds,meanvalues,spamcount,hamcount):
    count = 0
    j = 0
    feature_probablities = {}
    while j < 57:
        caseA = 0
        caseB = 0
        caseC = 0
        caseD = 0
        for fold in folds:
            if count == excludedfold:
                count =count +1
                continue
            else:
                count =count +1
                for email in fold:
                    if float(email[j]) < meanvalues[j]:
                        if int(email[57]) ==1:
                            caseA = caseA + 1
                        else:
                            caseC = caseC + 1
                    else:
                        if int(email[57]) ==1:
                            caseB = caseB + 1
                        else:
                            caseD = caseD + 1
        probablities = []
        probablities.append((float(caseA)+1)/(spamcount+2))
        probablities.append((float(caseB)+1)/(spamcount+2))
        probablities.append((float(caseC)+1)/(hamcount+2))
        probablities.append((float(caseD)+1)/(hamcount+2))
        feature_probablities[j] = probablities
        j = j+1
    
    return feature_probablities
                            
                    
#===============================================================================
# calculateMeanValues
#===============================================================================
def calculateMeanValues(i,folds):
    count =0
    feature_mean = {}
    i=0
    spamcount= 0
    hamcount = 0
    while i<58:
        feature_mean[i]=0.0
        i = i +1
        
    email_count = 0
    for fold in folds:
        if count == i:
            count += 1
            continue
        else:
            count += 1
            for email in fold:
                email_count = email_count + 1
                feature_count =0
                for feature in email:
                    feature_mean[feature_count] = feature_mean[feature_count] + float(feature)
                    if feature_count ==57:
                        if int(email[feature_count]) == 1:
                            spamcount +=1
                        else:
                            hamcount += 1
                    feature_count = feature_count + 1
    
   
    key = 0 
    while key<len(feature_mean):
            feature_mean[key] = float(feature_mean[key])/email_count
            key = key + 1
            
    return feature_mean,spamcount,hamcount
                
#===============================================================================
# predictSpam
#===============================================================================
def predictSpam(probablities, testing_fold, meanvalues,spamcount,hamcount):
    bernoulli_prediction =[] 
    bernoulli_result = []
    for email in testing_fold:
        count = 0
        sump = 0
        finalvalue = 0
        while count < 57:
            plist=probablities[count]
            if float(email[count]) <= meanvalues[count]:
                sump +=  math.log(float(plist[0])/float(plist[2]))
            else:
                sump +=  math.log(float(plist[1])/float(plist[3]))
            count = count + 1
        finalvalue = math.log(float(spamcount)/float(hamcount)) + sump
        bernoulli_result.append(finalvalue)
        if finalvalue > 0:
            bernoulli_prediction.append(1)
        else:
            bernoulli_prediction.append(0)
    return  bernoulli_prediction, bernoulli_result


        
#***************************************************************************************************************************** 
#Gaussian Random Variable
#***************************************************************************************************************************** 
 
#===============================================================================
# calculateClassConditionalMean
#===============================================================================
def calculateClassConditionalMean(i,folds,spamcount,hamcount):
    count =0
    feature_mean_spam = {}
    feature_mean_ham = {}
    i=0
    
    #initialize dictionary to 0.0
    while i<58:
        feature_mean_spam[i]=0.0
        feature_mean_ham[i]=0.0
        i = i +1
        
    j=0
    while j < 57:
        for fold in folds:
            if count == i:
                count += 1
                continue
            else:
                count += 1
                for email in fold:
                    if int(email[57]) == 1:
                        feature_mean_spam[j] = feature_mean_spam[j] + float(email[j])
                    else:
                        feature_mean_ham[j] = feature_mean_ham[j] + float(email[j])
        j += 1
                    
    
   
    key = 0 
    while key<len(feature_mean_spam):
            feature_mean_spam[key] = (float(feature_mean_spam[key])+1)/(spamcount+2)
            feature_mean_ham[key] = (float(feature_mean_ham[key])+1)/(hamcount+2)
            key = key + 1
            

            
    return feature_mean_spam,feature_mean_ham


#===============================================================================
# calculateGaussianVariance
#===============================================================================
def calculateGaussianVariance(i,folds,feature_mean,feature_mean_spam,feature_mean_ham,spamcount,hamcount):
    count = 0
    j=0
    feature_variance ={}
    feature_variance_spam ={}
    feature_variance_ham ={}
    while j < 57:
        feature_variance[j] = 0.0
        feature_variance_spam[j] = 0.0
        feature_variance_ham[j] = 0.0
        for fold in folds:
            if count == i:
                count =count +1
                continue
            else:
                count =count +1
                for email in fold:
                    feature_variance[j] += (float(email[j])-feature_mean[j]) * (float(email[j])-feature_mean[j])
                    if int(email[57])==1:
                        feature_variance_spam[j] += (float(email[j])-feature_mean_spam[j]) * (float(email[j])-feature_mean_spam[j])
                    else:
                        feature_variance_ham[j] += (float(email[j])-feature_mean_ham[j]) * (float(email[j])-feature_mean_ham[j])
        j += 1
        
    for feature in range(0,57):
        feature_variance_spam[feature] = (float(feature_variance_spam[feature]))/((float(spamcount)-1))
        feature_variance_ham[feature] = (float(feature_variance_ham[feature]))/((float(hamcount)-1))
        feature_variance[feature] /= (float(spamcount+ hamcount)-1)
        if (feature_variance_spam[feature] == 0 and feature_variance_ham[feature] == 0 and feature_variance[feature] == 0):
            feature_variance_spam[feature] = 1 
            feature_variance_ham[feature] = 1
            feature_variance[feature] = 1
        else:
            feature_variance_spam[feature] = (spamcount/(spamcount + 2))*feature_variance_spam[feature] + (2/(spamcount + 2))*feature_variance[feature]
            feature_variance_ham[feature] = (hamcount/(hamcount + 2))*feature_variance_ham[feature] + (2/(hamcount + 2))*feature_variance[feature]
    
    return feature_variance_spam,feature_variance_ham          
                        
#===============================================================================
# predictSpamGaussian
#===============================================================================
def predictSpamGaussian(testing_fold,feature_mean_spam,feature_mean_ham,feature_variance_spam,feature_variance_ham, spamcount, hamcount):
    gaussian_prediction =[]
    gaussian_result = []
    for email in testing_fold:
        finalvalue = 0.0
        tot_probablity =0.0
        feature = 0
        while feature in range(0,57):
            spam_probablity = 0.0
            ham_probablity = 0.0
            spam_probablity = ((float(email[feature]) - feature_mean_spam[feature]) * (float(email[feature]) - feature_mean_spam[feature])/(2*feature_variance_spam[feature]))+math.log(math.sqrt(feature_variance_spam[feature]))
            ham_probablity = ((float(email[feature]) - feature_mean_ham[feature]) * (float(email[feature]) - feature_mean_ham[feature])/(2*feature_variance_ham[feature]))+math.log(math.sqrt(feature_variance_ham[feature]))
            tot_probablity += ham_probablity - spam_probablity
            feature += 1
        finalvalue = math.log(spamcount/hamcount) + tot_probablity
        gaussian_result.append(finalvalue)
        if (finalvalue >= 0):
            gaussian_prediction.append(1)
        else:
            gaussian_prediction.append(0)
        
    return gaussian_prediction , gaussian_result
        
#***************************************************************************************************************************** 
# Feature Distribution using Histogram
#***************************************************************************************************************************** 
 
#===============================================================================
# calculateMinMaxValues
#===============================================================================
def calculateMinMaxValues(excludedfold,folds):
    i = 0
    count = 0
    min_value = 0
    max_value = 0
    min_max = {}
    while i<57:
        thresholds = []
        for fold in folds:
            if count == excludedfold:
                count =count +1
                continue
            else:
                count =count +1
            for email in fold:
                if float(email[i]) <= min_value:
                    min_value = float(email[i])
                if float(email[i]) > max_value:
                    max_value = float(email[i])
        thresholds.append(min_value)
        thresholds.append(max_value)
        min_max[i] = thresholds
        i += 1
    return min_max

#===============================================================================
# calculateThresholdValues
#===============================================================================
def calculateThresholdValues(excudedfold,folds,feature_mean_spam,feature_mean_ham, feature_mean ):
    
    feature_thresholds = {}
    thresholds = []
    min_max = calculateMinMaxValues(excudedfold,folds)
    #feature_max = []
    feature =0
    for feature in range(0,57):
        thresholds = min_max[feature]
        if feature_mean_spam[feature] <= feature_mean[feature]:
            thresholds.append(feature_mean_spam[feature])
            thresholds.append(feature_mean[feature])
            thresholds.append(feature_mean_ham[feature])
        else:
            thresholds.append(feature_mean_ham[feature])
            thresholds.append(feature_mean[feature])
            thresholds.append(feature_mean_spam[feature])
        feature_thresholds[feature]=thresholds 
    
    return feature_thresholds

#===============================================================================
# calculateHistogramProbablities
#===============================================================================
def calculateHistogramProbablities (excludedfold,folds,feature_thresholds,spamcount,hamcount):
    count = 0
    j = 0
    feature_probablities = {}
    while j < 57:
        bucket1_spam = 0
        bucket2_spam = 0
        bucket3_spam = 0
        bucket4_spam = 0
        bucket1_ham = 0
        bucket2_ham = 0
        bucket3_ham = 0
        bucket4_ham = 0
        thresholds = feature_thresholds[j]
        for fold in folds:
            if count == excludedfold:
                count =count +1
                continue
            else:
                count =count +1
                for email in fold:
                    if int(email[57]) ==1:
                        if float(email[j]) > thresholds[0] and float(email[j])<= thresholds[2]:
                            bucket1_spam += 1
                        elif float(email[j]) > thresholds[2] and float(email[j]) <= thresholds[3]:
                            bucket2_spam += 1
                        elif float(email[j]) > thresholds[3] and float(email[j]) <= thresholds[4]:
                            bucket3_spam += 1
                        elif float(email[j]) > thresholds[4] and float(email[j]) <= thresholds[1]:
                            bucket4_spam += 1
                    else:
                        if float(email[j]) > thresholds[0] and float(email[j])<= thresholds[2]:
                            bucket1_ham += 1
                        elif float(email[j]) > thresholds[2] and float(email[j]) <= thresholds[3]:
                            bucket2_ham += 1
                        elif float(email[j]) > thresholds[3] and float(email[j]) <= thresholds[4]:
                            bucket3_ham += 1
                        elif float(email[j]) > thresholds[4] and float(email[j]) <= thresholds[1]:
                            bucket4_ham += 1
        probablities = []
        probablities.append((float(bucket1_spam)+1)/(spamcount+2))
        probablities.append((float(bucket2_spam)+1)/(spamcount+2))
        probablities.append((float(bucket3_spam)+1)/(hamcount+2))
        probablities.append((float(bucket4_spam)+1)/(hamcount+2))
        probablities.append((float(bucket1_ham)+1)/(spamcount+2))
        probablities.append((float(bucket2_ham)+1)/(spamcount+2))
        probablities.append((float(bucket3_ham)+1)/(hamcount+2))
        probablities.append((float(bucket4_ham)+1)/(hamcount+2))
        feature_probablities[j] = probablities
        j = j+1
    
    return feature_probablities
        
#===============================================================================
# predictSpamHistogram
#===============================================================================
def predictSpamHistogram(feature_probablities, testing_fold, feature_thresholds,spamcount,hamcount):
    histogram_prediction =[] 
    histogram_result = []
    for email in testing_fold:
        count = 0
        sump = 0
        finalvalue = 0
        while count < 57:
            plist=feature_probablities[count]
            thresholds = feature_thresholds[count]
            if float(email[count]) > thresholds[0] and float(email[count])<= thresholds[2]:
                sump +=  math.log(float(plist[0])/float(plist[4]))
            elif float(email[count]) > thresholds[2] and float(email[count]) <= thresholds[3]:
                sump +=  math.log(float(plist[1])/float(plist[5]))
            elif float(email[count]) > thresholds[3] and float(email[count]) <= thresholds[4]:
                sump +=  math.log(float(plist[2])/float(plist[6]))
            elif float(email[count]) > thresholds[4] and float(email[count]) <= thresholds[1]:
                sump +=  math.log(float(plist[3])/float(plist[7]))
            count = count + 1
        finalvalue = math.log(float(spamcount)/float(hamcount)) + sump
        histogram_result.append(finalvalue)
        if finalvalue > 0:
            histogram_prediction.append(1)
        else:
            histogram_prediction.append(0)
   
    return  histogram_prediction , histogram_result  
        
        
                                             
                    
    
    
    
 
 
#*****************************************************************************************************************************
#*****************************************************************************************************************************
   

def main():
    folds = readKFolds()
    i=0
    option = int(input("Enter the Clsssifier you want to use:- \n 1. Naive-Bayes using Bernoulli Random Variable Model \n 2. Naive-Bayes using Gaussian Random Variable  \n 3. Naive-Bayes using Histogram \n 4. All of these \n Enter Your Choice "))
    if option != 1 and option!=2 and option!=3 and option!=4:
         print("Invalid Input")
         
    while i<10:
        if option == 1:
            feature_mean,spamcount,hamcount = calculateMeanValues(i,folds)
            feature_probablities = calculateProbablities(i,folds,feature_mean,spamcount,hamcount)
            result,bernoulli_result = predictSpam(feature_probablities,folds[i],feature_mean,spamcount,hamcount)
            act_result = getActualResult(folds[i])
            if i == 0:
                genrateROC(bernoulli_result, act_result)
                plt.show()
                
            print("Error Rate = %s ,False Positive = %s ,False Negative = %s, FPR = %s , FNR = %s  "  %calculateErrorRate(result,act_result))
            print('\n')
        elif option == 2:
            feature_mean,spamcount,hamcount = calculateMeanValues(i,folds)
            feature_mean_spam,feature_mean_ham = calculateClassConditionalMean(i,folds,spamcount,hamcount)
            feature_variance_spam,feature_variance_ham = calculateGaussianVariance(i,folds,feature_mean,feature_mean_spam,feature_mean_ham,spamcount,hamcount)
            result,gaussian_result =predictSpamGaussian(folds[i],feature_mean_spam,feature_mean_ham,feature_variance_spam,feature_variance_ham, spamcount, hamcount)
            act_result = getActualResult(folds[i])
            if i == 0:
                genrateROC(gaussian_result, act_result)
                plt.show()
            print("Error Rate = %s ,False Positive = %s ,False Negative = %s, FPR = %s , FNR = %s  "  %calculateErrorRate(result,act_result))
            print('\n')
        elif option == 3:
            feature_mean,spamcount,hamcount = calculateMeanValues(i,folds)
            feature_mean_spam,feature_mean_ham = calculateClassConditionalMean(i,folds,spamcount,hamcount)
            feature_thresholds = calculateThresholdValues(i,folds,feature_mean_spam,feature_mean_ham, feature_mean)
            feature_probablities = calculateHistogramProbablities (i,folds,feature_thresholds,spamcount,hamcount)
            result,histogram_result = predictSpamHistogram(feature_probablities,folds[i], feature_thresholds,spamcount,hamcount)
            act_result = getActualResult(folds[i])
            if i == 0:
                genrateROC(histogram_result, act_result)
                plt.show()
            print("Error Rate = %s ,False Positive = %s ,False Negative = %s, FPR = %s , FNR = %s  "  %calculateErrorRate(result,act_result))
            print('\n')
        elif option == 4:
            feature_mean,spamcount,hamcount = calculateMeanValues(i,folds)
            feature_mean_spam,feature_mean_ham = calculateClassConditionalMean(i,folds,spamcount,hamcount)
            
            feature_probablities = calculateProbablities(i,folds,feature_mean,spamcount,hamcount)
            result,bernoulli_result = predictSpam(feature_probablities,folds[i],feature_mean,spamcount,hamcount)
            
            feature_variance_spam,feature_variance_ham = calculateGaussianVariance(i,folds,feature_mean,feature_mean_spam,feature_mean_ham,spamcount,hamcount)
            result,gaussian_result =predictSpamGaussian(folds[i],feature_mean_spam,feature_mean_ham,feature_variance_spam,feature_variance_ham, spamcount, hamcount)
            
            feature_thresholds = calculateThresholdValues(i,folds,feature_mean_spam,feature_mean_ham, feature_mean)
            feature_probablities = calculateHistogramProbablities (i,folds,feature_thresholds,spamcount,hamcount)
            result,histogram_result = predictSpamHistogram(feature_probablities,folds[i], feature_thresholds,spamcount,hamcount)
            
            act_result = getActualResult(folds[i])
            if i == 0:
                genrateROC(bernoulli_result, act_result)
                genrateROC(gaussian_result, act_result)
                genrateROC(histogram_result, act_result)
                plt.show()
            print("Error Rate = %s ,False Positive = %s ,False Negative = %s, FPR = %s , FNR = %s  "  %calculateErrorRate(result,act_result))
            print('\n')
        else:
            break
        i = i+1
       
    
if __name__=="__main__":
    main()