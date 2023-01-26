import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("BEE529 Dataset BonnevilleChinook_2010to2015.csv")

print(df.head())

#scrub
print("Do we have any NaN Values:", df.isnull().values.any()) #do we have any nan values -- yes
print("How many NaN values do we Have:", df.isnull().sum().sum()) #how many -- 147

# we have 147 NaN values which means at most we could have 147 rows with at least 1 NaN value. Since we have 2197 rows of data
# I feel comfortable with removing all rows with NaN and retaining the integrity of the data. 

df.dropna(inplace=True) # remove rows with at least 1 NaN

#assign columns as variables
count = df["Unnamed: 2"].values
outflow = df["Unnamed: 3"].values
temp = df["Unnamed: 4"].values
turbidity = df["Unnamed: 5"].values

# remove index 0 which is a column string
count = np.delete(count,0)
outflow = np.delete(outflow, 0)
temp = np.delete(temp,0)
turbidity = np.delete(turbidity,0)

# need to convert entries in numpy array to float.  Pandas defaults to importing all values as strings when hitting a NaN. This is an computationally inefficient way to 
# convert the arrays especially if we were using a larger data sets. If they were larger I may important to pandas, understand the data and then re-read the data 
# without NaN so pandas imports values as int and I wouldn't need to iterate across array again do this second conversion.

count = count.astype(np.float64)
outflow = outflow.astype(np.float64)
temp = temp.astype(np.float64)
turbidity = turbidity.astype(np.float64)


# visualization and regression model class for research area of interest
class SalmonDamModel():
    """input takes multiple lists, first element of each list is string of variable name, second element is array object"""
    
    def __init__(self, *args) -> None:
        """parse through input lists and initialize object for each list. each object is a list with string variable name and units as index 0 and corresponding numpy array as value. This also inits dictionary for regression variables"""
        for j in args: 
            setattr(self,j[0],[j[0],j[1]])

        # simple regression dictionary
        self.simp_regs_dict = {"slope": None, "intercept": None, "r2value": None, "pvalue": None, "stderr": None}

        # simple regression model
        self.simp_model = None
        
        # multi-regression parameters est
        self.est = None
        self.est_params = None
        self.est_rsquared = None

        # multi-regression model
        self.multi_model = None

        # units dictionary for making graphs
        self.units = {"count": " (fish/day)", "temp": " ($^\circ$C)", "outflow": " (kcfs)", "turbidity": " (secchi ft)"}


        if len(args) > 2: #populate xObs array
            temp = []
            for i in args: 
                if i[0] != "count":
                    temp.append(i[1])
            

            temp2 = np.array(temp).T  # make numpy array and transpose for multi regression
            self.xObsArray = temp2

    
    def simpVis (self,xObs,yObs) -> None:
        """input should be two class object (e.g. simpVis(<instance>.outflow, <instance>.count)) with x var first and y var second. This will display simple scatter plot of both inputs"""
        
        plt.plot(xObs[1],yObs[1],'o', alpha = .1)
        plt.ylabel(yObs[0] + self.units[yObs[0]])
        plt.xlabel(xObs[0] + self.units[xObs[0]])
        #plt.title(xObs[0] + " " + "vs" + " " + yObs[0])
        plt.show()

    def regresVis(self, xObs, yObs, model) -> None:
        """input should be three instance objects, for input observations, output observations, and regression model (self.simp_model). This will display a plot with x input in y input and overlay modeled regression"""
        
        plt.plot(xObs[1],yObs[1],'o', color='blue' )   # add observation series
        plt.plot(xObs[1], model, '-', color='red' )    # add modeled series
        print(self.simp_regs_dict["r2value"])
        plt.legend(['Observed', 'Predicted'], title ="r2=%.4f"%self.simp_regs_dict["r2value"], loc='upper center')
        plt.xlabel(xObs[0] + self.units[xObs[0]])
        plt.ylabel( yObs[0] + self.units[yObs[0]])
        #plt.text(0.01, 0.99, , fontsize=10 )
        plt.show()
        
    
    def simpRegres(self, xObs, yObs) -> None:
        """input should be two class object (e.g. simpVis(<instance>.outflow, <instance>.count)) with x var first and y var second. This will preform simple linear regression on inputs and update master regression dictionary"""
        
        self.simp_regs_dict["slope"], \
        self.simp_regs_dict["intercept"], \
        self.simp_regs_dict["r2value"], \
        self.simp_regs_dict["pvalue"], \
        self.simp_regs_dict["stderr"] = stats.linregress(xObs[1], yObs[1])

        # make r value a r2 value
        self.simp_regs_dict["r2value"] = self.simp_regs_dict["r2value"]**2

    def simpModel(self, regres, xObs) -> None:
        """First argument will be object reference to regression model dict (e.g. <instance>.simp_regs_dic), second argument will be object reference to x observations values for model (e.g. <instance>.turbidity). Makes linear regression model"""
        
        self.simp_model= regres["slope"]*(xObs[1]) + regres["intercept"]

    def multi_regs_and_model(self, yObs) -> None:
        """input should be one class object -- dependent variable (e.g. <instance>.count)). This makes a multi regression model"""

        # perform OLS
        self.xObsArray = sm.add_constant(self.xObsArray)
        self.multi_model = sm.OLS(yObs[1], self.xObsArray)
        self.est = self.multi_model.fit()
        self.est_params = self.est.params
        self.est_rsquared = self.est.rsquared

    def observed_predicted_vis(self, yObs):
        """input should be one class object -- dependent variable (e.g. <instance>.count)). Displays a observed vs predicted plot for multi regression"""

        yModeled = self.est.predict(self.xObsArray)

        plt.plot(yObs[1],yModeled, 'o', color='blue')    # add modeled series (and add alpha to show the density of points)
        plt.plot( [0,70000], [0,7000], '-', color='red' )     # add line of perfect fit modeled series (1:1 line)
        plt.title( "Regression Results")
        plt.xlabel( "Observed")
        plt.ylabel( "Predicted")
        plt.legend(['Observed', 'Perfect fit modeled series'], title ="r2=%.4f"%self.est_rsquared, loc='lower right')
        plt.show()


#init four hypothesis
hypotheses1 = SalmonDamModel(["count", count], ["outflow",outflow])
hypotheses2 = SalmonDamModel(["count", count], ["temp",temp])
hypotheses3 = SalmonDamModel(["count", count], ["turbidity",turbidity])
hypotheses4 = SalmonDamModel(["count", count], ["outflow",outflow], ["temp",temp],["turbidity",turbidity])

#run hypothesis 1
hypotheses1.simpVis(hypotheses1.outflow, hypotheses1.count)
hypotheses1.simpRegres(hypotheses1.outflow, hypotheses1.count)
hypotheses1.simpModel(hypotheses1.simp_regs_dict, hypotheses1.outflow)
hypotheses1.regresVis(hypotheses1.outflow, hypotheses1.count,hypotheses1.simp_model)
print('Regression results of Hypothesis 1:', hypotheses1.simp_regs_dict)

#run hypothesis 2
hypotheses2.simpVis(hypotheses2.temp, hypotheses2.count)
hypotheses2.simpRegres(hypotheses2.temp, hypotheses2.count)
hypotheses2.simpModel(hypotheses2.simp_regs_dict, hypotheses2.temp)
hypotheses2.regresVis(hypotheses2.temp, hypotheses2.count,hypotheses2.simp_model)
print('Regression results of Hypothesis 2:', hypotheses2.simp_regs_dict)

#run hypothesis 3
hypotheses3.simpVis(hypotheses3.turbidity, hypotheses3.count)
hypotheses3.simpRegres(hypotheses3.turbidity, hypotheses3.count)
hypotheses3.simpModel(hypotheses3.simp_regs_dict, hypotheses3.turbidity)
hypotheses3.regresVis(hypotheses3.turbidity, hypotheses3.count,hypotheses3.simp_model)
print('Regression results of Hypothesis 3:', hypotheses3.simp_regs_dict)

#run hypothesis4
hypotheses4.multi_regs_and_model(hypotheses4.count)
hypotheses4.observed_predicted_vis(hypotheses4.count)
print(hypotheses4.est.summary())


