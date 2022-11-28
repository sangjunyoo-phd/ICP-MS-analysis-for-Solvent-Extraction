import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy import stats

plt.rcParams['font.size'] = 8
rcParams['mathtext.default'] = 'regular'
pd.options.mode.chained_assignment = None  # default='warn'

# =============================================================================
#
# DATA ACQUISITION
#
# =============================================================================

# Data file: csv format with 14 columns
# First Column: Sample label 
#   starts with element name (Er or Nd) or Mixture (Er and Nd)
#   surfactant concentration, mixing time comes next but random order
#   surfactant concentration starts with "r="
#   Mixing time sometimes only contain number or ends with "min"
# 2~7th Column: Nd mass spectroscopy data
#   Isotope 142~148 (no 147) analyzed
#   concentration in ppb (float)
# 8~10th Column: Er mass spectroscopy data
#   Isotope 166~168 Analyzed
#   concentration in ppb (float)
# 11, 12th Column: Averaged Nd and Er concentration in ppb
#   These columns are not needed since Dilution factor should be considered first & I am interested in M unit
# 13, 14th Column: Dilution
#   Extracted sample is diluted with 3% HNO3
#   extracted: extracted sample mass in g
#   HNO3: mass of 3% HNO3 added for dilution in g

# =============================================================================
# 
# GOAL
# 
# =============================================================================

# Data Cleaning
# Decouple components in the "Sample" column:
#   Sample type ("Pure", "Mixture"): pure solution contains only one element, whereas mixture has both
#   element ("Er","Nd")
#   surfactant concentration: "r=#" (as a string to make plotting stage easier)
#   Mixing time: in integer or float64

# Data Analysis
# The numbers in data: remaining (unextracted) ion concentration -> switch it to "Extracted" concentration
# extracted concentration = 1-unextracted/initial (x100%)
# Get 1. extracted concentration, 2. extracted percentage
# Check the reaction order: Plot log(c/c0) and c0/c with linear regression, Manually check the linearity
#   If linear in log plot: 1st order reaction c = c0*exp(-kt)
#   If linear in 1/c plot: 2nd order reaction dc/dt ~ c^2
# 
# Note!!!
# In each data file (csv file) there are stock solution entries
# The concentrations in a csv file should be analyzied with the stock solution concenration in the "SAME FILE"

def read_label(df):
    ''' 
    Read sample label column (1st column of df) and return
    1. Sample Type: Pure or Mixture
    2. Element: Er or Nd -> Er or Nd, Mix or Mixture -> Er and Nd
    3. r: DHDP/Ln
    4. t: Vortexing time 
    '''
    # Label starts with: Mix, Mixture, Er, or Nd
    # r or t label comes next: r always starts with "r=" and t either have or have not t= or unit(min or mins)
    
    # Cleaning: no spaces and underscores
    df.columns = df.columns.str.replace(" ","") # Remove empty spaces in column names
    df['Sample'] = df['Sample'].str.replace(" ","_") 
    df['Sample'] = df['Sample'].str.replace("_min","min")
    
    # Sample Type converter: read Sample Column and return Mixture or Pure
    def samtype(row):
        if row["Sample"].split('_')[0] == 'Mix' or row["Sample"].split('_')[0] == 'Mixture':
            return "Mixture"
        elif row["Sample"].split('_')[0] == 'Er':
            return "Pure"
        elif row["Sample"].split('_')[0] == 'Nd':
            return "Pure"
        else:
            print("Error in Sample type: Check the element name")
    
    # Sample r and t
    def samrt(row):
        if 'r=' in row["Sample"]:
            for strs in row["Sample"].split('_')[1:]:
                if strs.startswith('r='):
                    r_value = strs
                else: # if not starts with 'r=': information about t!
                    t_value = strs
                    if t_value.startswith('t='):
                        t_value = t_value.replace('t=','')
                    if t_value.endswith('min'):
                        t_value = t_value.replace('min','')
            return (r_value, int(t_value))
        else:
            return ('stock',0)
    
    df["sample_type"] = df.apply(lambda row: samtype(row), axis=1)
    df["r"]=df.apply(lambda row: samrt(row)[0], axis=1)
    df["mixing time"] = df.apply(lambda row: samrt(row)[1], axis=1)
    
    return df
    
def ms_to_conc(file_name):
    """
    Converting Mass Spectrometry data to remaining concentration
    return a DataFrame with columns=["sample_type",'r',"mixing time",'Er_conc','Nd_conc']
    """
     # Return or generate global variables of Extraction result in concentration
    path = f'{file_name}/'
    files = os.listdir(path)    # List of csv file names in the path
    df_final = pd.DataFrame(columns=["sample_type",'r',"mixing time",'Er_conc','Nd_conc'])
    for file in files:
        if file.endswith('.csv') == True:   # Read csv files only
            # Make sample type, element, r columns based on "Sample" values using the function defined above
            df = pd.read_csv(path+file,skiprows=1,index_col=False)
            df = read_label(df)
            
            # Convert the ICP-MS results to the remaining concentrations by taking average         
            df["dilute"]=(df['extracted']+df['HNO3'])/df['extracted']
            def get_Er_conc(row):
                if row["Sample"].split('_')[0] == 'Er' or row["Sample"].split('_')[0] == 'Mix' or row["Sample"].split('_')[0] == 'Mixture':
                    return row['dilute']*(row['166Er']/166+row['167Er']/167+row['168Er']/168)/3
            def get_Nd_conc(row):
                if row["Sample"].split('_')[0] == 'Nd' or row["Sample"].split('_')[0] == 'Mix' or row["Sample"].split('_')[0] == 'Mixture':
                    return row['dilute']*(row['142Nd']/142+row['143Nd']/143+row['145Nd']/145+row['146Nd']/146+row['148Nd']/148)/6
            df['Er_conc'] = df.apply(lambda row:get_Er_conc(row),axis=1)
            df['Nd_conc'] = df.apply(lambda row:get_Nd_conc(row),axis=1)            
            df = df[["sample_type",'r',"mixing time",'Er_conc','Nd_conc']]
            
            list_r_df = [] # List of unique r values without 'stock' in a single exp file
            for r_val in df['r'].unique():
                if r_val == 'stock':
                    pass
                else:
                    list_r_df.append(r_val)
            
            temp = df[df['r']=='stock']
            
            if len(list_r_df)==1:
                temp1 = temp
                for j in range(len(temp1)):
                    temp1['r'].iloc[j] = list_r_df[0]
                    
            for i in range(len(list_r_df)):
                temp_i = temp.replace({'stock':list_r_df[i]})
                df = pd.concat([df,temp_i],axis=0)
            
            df_final = pd.concat([df_final,df],axis=0,ignore_index=True)

    df_final = df_final.drop(df_final[df_final['r']=='stock'].index)
    df_final.sort_values(by=['sample_type','r','mixing time'],inplace=True,axis=0,ignore_index=True)
    #df_final.to_csv('merged_df.csv', index=False)
    return df_final

def call_data(df, sam_type, element, r):
    # Passing string of type element r
    # Return the extracted percentage as array
    temp = pd.DataFrame(columns=['time','conc','perc'])
    time = df[(df['sample_type']==sam_type)&(df['r']==r)&(df[f'{element}_conc'].notnull())]['mixing time']
    conc = df[(df['sample_type']==sam_type)&(df['r']==r)&(df[f'{element}_conc'].notnull())][f'{element}_conc']
    time = time.astype('float')
    
    temp['time']=time
    temp['conc']=conc
    
    stock_conc=temp[temp['time']==0]['conc'].values
    temp['perc'] = 100 * (stock_conc - temp['conc'])/stock_conc
    
    return time.values, temp['perc'].values

def call_conc(df, sam_type, element, r):
    # Passing string of type element r
    # Return the mixing time and extracted concentrations as tuple: directly use to fit the data
    time = df[(df['sample_type']==sam_type)&(df['r']==r)&(df[f'{element}_conc'].notnull())]['mixing time']
    conc = df[(df['sample_type']==sam_type)&(df['r']==r)&(df[f'{element}_conc'].notnull())][f'{element}_conc']
    time = time.astype('float')
    return time.values, conc.values


# =============================================================================
# 
# DATA ANALYSIS PART
# 
# =============================================================================

def curve_peleg(x,k1,k2):   # Langmuir isotherm adsorption
    return x/(1/k1+x/k2)

def curve_exp(x,c,k):   # 1st order reaction
    return c*(1-np.exp(-x*k))

def curve_exp_modified(x,c,k,b):    # 1st order reaction with base extraction
    return c*(1-np.exp(b-x*k))

def curve_2exp(x,c_1,k_1,c_2,k_2):  # 1st order reaction with two mechanism
    return c_1*(1-np.exp(-x*k_1)) + c_2*(1-np.exp(-x*k_2))

def fit_exp(t,data):
    x=np.linspace(0,45,100)
    data_fit=data[1:]
    t_fit=t[1:]
    err=2**0.5 * (1-0.01*np.array(data_fit))
    popt, pcov= curve_fit(curve_exp,t_fit,data_fit,sigma=err,maxfev=10000,
                          bounds=([max(data),0],[100,np.inf]))
    y=curve_exp(x, *popt)
    return x,y, popt,pcov

def fit_exp_minmax(t,data,k_range):
    x=np.linspace(0,45,100)
    data_fit=data[1:]
    t_fit=t[1:]
    err=2**0.5 * (1-0.01*np.array(data_fit))
    popt, pcov= curve_fit(curve_exp,t_fit,data_fit,sigma=err,maxfev=10000,
                          bounds=([0,k_range[0]],[100,k_range[1]]))
    y=curve_exp(x, *popt)
    return x,y, popt,pcov

def fit_exp_modified(t,data):
    x=np.linspace(0,45,100)
    data_fit=data[1:]
    t_fit=t[1:]
    err=2**0.5 * (1-0.01*np.array(data_fit))
    popt, pcov= curve_fit(curve_exp_modified,t_fit,data_fit,sigma=err,maxfev=10000,
                          bounds=([0,-np.inf,-np.inf],[100,np.inf,np.inf]))
    y=curve_exp_modified(x, *popt)
    return x,y, popt,pcov

def get_chisq(t, data, model, popt):
    # Return Chi square values to automate the model selection
    t_fit=t[1:]
    data_fit=data[1:]
    t_fit=np.asarray(t_fit)
    data_fit = np.asarray(data_fit)
    err=2**0.5*(1-0.01*np.asarray(data_fit))
    
    if str(model)=='1exp':
        y=curve_exp(t_fit,*popt)
    elif str(model)=='2exp':
        y=curve_2exp(t_fit,*popt)
    elif str(model)=='1exp_modified':
        y=curve_exp_modified(t_fit,*popt)
    chisq = (data_fit-y)**2/err**2
    return sum(chisq)

def get_params(df):
    """
    Get fitted parameters Including:
        1. Equilibrium E (%)
        2. kinetic constant (k) (min^-1)
        3. b term (if exists)
    """
    
    # Initialize
    sample_=list()
    element_=list()
    r_=list()
    E_eq_=list()
    E_eq_modified_=list()
    k_=list()
    k_modified_=list()
    b_=list()
    
    for sam_type in ['Pure']:
        if sam_type == 'Pure':
            r_cand = ['r=1', 'r=3', 'r=4.5', 'r=6']
        elif sam_type == 'Mixture':
            r_cand = ['r=1', 'r=3', 'r=6']
        for element in ['Er', 'Nd']:
            for r in r_cand:
                sample_.append(sam_type)
                element_.append(element)
                r_.append(r)
                
                t, E = call_data(df, sam_type, element, r)
                x,y,popt,pcov = fit_exp(t,E)
                E_eq_.append(popt[0])
                k_.append(popt[1])
                
                x,y,popt,pcov = fit_exp_modified(t,E)
                E_eq_modified_.append(popt[0])
                k_modified_.append(popt[1])
                b_.append(popt[2])
                
    exp_dict = {'sample':sample_, 'element':element_, 'r':r_, 'E_eq':E_eq_,'k':k_}
    df_exp = pd.DataFrame(data=exp_dict, columns=exp_dict.keys())
    df_exp.to_csv('exp_fit.csv', index=False)
    
    exp_modified_dict = {'sample':sample_, 'element':element_, 'r':r_, 'E_eq':E_eq_modified_,'k':k_modified_, 'b':b_}
    df_exp_modified = pd.DataFrame(data=exp_modified_dict, columns=exp_modified_dict.keys())
    df_exp_modified.to_csv('exp_modified.csv', index=False)

def plot_analysis_without_b(df):
    """
    Visualize the curve fit of E=Eeq*(1-exp(-k*t)) for analysis
    NOT A FIGURE FOR PUBLICATION
    """
    r_cand = ['r=3','r=4.5','r=6']
    for sam_type in ['Pure']:
        for element in ['Er','Nd']:
            for r in r_cand:
                # Make figure instance
                fig = plt.figure()
                fig_width=3.25
                aspect_ratio = 2.0
                fig_height=aspect_ratio*3.25/1.618
                fig.set_size_inches(fig_width,fig_height)
                
                # Set Axis
                # ax1: Bottm: Log(1-E/Eeq) and linear fit plot
                ax1=fig.add_axes([0.2,0.2/aspect_ratio,0.7,0.7/aspect_ratio], xlim = (-1.5,45))
                # ax2: Top Extraction result and exponential fit
                ax2=fig.add_axes([0.2,1.1/aspect_ratio,0.7,0.7/aspect_ratio], xlim = (-1.5,45), ylim=(-5,105))
                ax1.set_xlabel('Mixing Time (min)')
                ax1.set_ylabel('log (1-E/Eeq)')
                ax2.set_ylabel('Extraction (%)')
                ax2.set_title(f'Without b {sam_type} {element} {r}')
                
                # Call Data
                t, E = call_data(df, sam_type, element, r)
                
                # Plot Extraction result
                ax2.plot(t,E, marker='o', linewidth=0, markerfacecolor='None', markeredgewidth=1)
                # Plot exponential fit
                x,y,popt,pcov=fit_exp(t,E)
                E_eq = popt[0]
                k = popt[1]
                ax2.plot(x,y,linestyle='--')
                ax2.axhline(E_eq,0,1,linestyle='--',linewidth=0.5,color='k')
                ax2.text(43,1, f'Eeq: {round(E_eq,2)}\nk: {round(k,2)}',
                         ha='right', va='bottom')
                
                # Plot log(1-E/E_eq)
                if r == 'r=3' and element=='Er':
                    t_temp=t[:-1]
                    E_temp=E[:-1]
                    ax1.plot(t_temp,np.log(1-E_temp/E_eq),linewidth=0,marker='o',markerfacecolor='None', markeredgewidth=1)
                else:
                    ax1.plot(t,np.log(1-E/E_eq),linewidth=0,marker='o',markerfacecolor='None', markeredgewidth=1)
                ax1.plot(np.linspace(0,43),-k*np.linspace(0,43), color='k', linestyle='--')
                
                plt.savefig(f'Analysis_plot/withoutb/{sam_type}_{element}_{r}.png', dpi=600)
                
def plot_analysis_with_b(df):
    """
    Visualize the curve fit of E=Eeq*(1-exp(-k*t+b)) for analysis
    NOT A FIGURE FOR PUBLICATION
    """
    r_cand = ['r=3','r=4.5','r=6']
    for sam_type in ['Pure']:
        for element in ['Er','Nd']:
            for r in r_cand:
                # Make figure instance
                fig = plt.figure()
                fig_width=3.25
                aspect_ratio = 2.0
                fig_height=aspect_ratio*3.25/1.618
                fig.set_size_inches(fig_width,fig_height)
                
                # Set Axis
                # ax1: Bottm: Log(1-E/Eeq) and linear fit plot
                ax1=fig.add_axes([0.2,0.2/aspect_ratio,0.7,0.7/aspect_ratio], xlim = (-1.5,45))
                # ax2: Top Extraction result and exponential fit
                ax2=fig.add_axes([0.2,1.1/aspect_ratio,0.7,0.7/aspect_ratio], xlim = (-1.5,45), ylim=(-5,105))
                ax1.set_xlabel('Mixing Time (min)')
                ax1.set_ylabel('log (1-E/Eeq)')
                ax2.set_ylabel('Extraction (%)')
                ax2.set_title(f'With b {sam_type} {element} {r}')
                
                # Call Data
                t, E = call_data(df, sam_type, element, r)
                
                # Plot Extraction result
                ax2.plot(t,E, marker='o', linewidth=0, markerfacecolor='None', markeredgewidth=1)
                # Plot exponential fit
                x,y,popt,pcov=fit_exp_modified(t,E)
                E_eq = popt[0]
                k = popt[1]
                b = popt[2]
                ax2.plot(x,y,linestyle='--')
                ax2.text(43,1, f'Eeq: {round(E_eq,2)}\nk: {round(k,3)}\nb: {round(b,3)}',
                         ha='right', va='bottom')
                
                # Plot log(1-E/E_eq)
                ax1.plot(t,np.log(1-E/E_eq),linewidth=0,marker='o',markerfacecolor='None', markeredgewidth=1)
                ax1.plot(np.linspace(0,43),-k*np.linspace(0,43)+b, color='k', linestyle='--')
                
                plt.savefig(f'Analysis_plot/withb/{sam_type}_{element}_{r}.png', dpi=600)
                

def log_plot(df):
    """
    To check if the reaction is 1st order (~exp(-k*t)) or not
    log of the remaining conc should be linear
    Check this with visualization
    """
    
    r_cand = ['r=3','r=4.5','r=6']
    k=[]
    b=[]
    E_eq_=[]
    sam=[]
    for sam_type in ['Pure']:#df['sample_type'].unique():
        for element in ['Er','Nd']:
            for r in r_cand:
                sam.append(f'{sam_type} {element} {r}')
                fig=plt.figure()
                fig_width=3.25
                aspect_ratio = 2.0
                fig_height=aspect_ratio*3.25/1.618
                fig.set_size_inches(fig_width,fig_height)
                
                # Set Axis
                t, E = call_data(df, sam_type, element, r)
                ax1=fig.add_axes([0.2,0.2/aspect_ratio,0.7,0.7/aspect_ratio], xlim = (-1.5,45))   # Bottom:Log plot
                ax2=fig.add_axes([0.2,1.1/aspect_ratio,0.7,0.7/aspect_ratio], xlim = (-1.5,45), ylim=(-5,105))  # Top exp
                ax1.set_xlabel('Mixing Time (min)')
                ax1.set_ylabel('log (1-E/Eeq)')
                ax2.set_ylabel('Extraction (%)')
                
                # Plot extraction (E) in ax2
                ax2.plot(t,E, linewidth=0, markersize=3, marker='o', markerfacecolor='None',markeredgewidth=1)
                x,y,popt,pcov=fit_exp(t,E)
                E_eq = popt[0]
                k_exp = popt[1]
                E_eq_.append(E_eq)
                
                ax2.plot(x,y, linewidth=1)
                ax2.axhline(E_eq,0,1,linewidth=0.5,linestyle='--',color='k')
                ax2.text(45,0,f'Eeq:{E_eq:.5f}\nk:{k_exp:.5f}',va='bottom',ha='right')
                
                inside_log = 1-E/E_eq
                
                
                if sam_type == 'Pure' and element == 'Er' and r == 'r=3':
                    # Plot the log (1-E/Eeq) except for the last point (outlier)
                    ax1.plot(t[:-1],np.log(inside_log[:-1].astype('float64')),linewidth=0,marker='o',markersize=3)
                    slope, intercept, rvalue, pvalue, std_err = stats.linregress(t, np.log(inside_log.astype('float64')))
                    
                    def one_line():
                        slope, intercept, rvalue, pvalue, std_err = stats.linregress(t[:-1], np.log(inside_log[:-1].astype('float64')))
                        x=np.linspace(0,45)
                        y=slope*x+intercept
                        ax1.plot(x,y,color='k',linewidth=1,linestyle='--')
                        ax1.text(44,0,f'y=k*x+b\nk={slope:.5f}\nb={intercept:.5f}\nR$^2$={rvalue**2:.5f}',va='top',ha='right')
                        # Extract Fitting parameters
                        k.append(abs(slope))
                        b.append(intercept)
                        
                    one_line()
                    
                else:
                    # Plot log(1-E/Eeq)
                    ax1.plot(t,np.log(inside_log.astype('float64')),linewidth=0,marker='o',markersize=3)
                    slope, intercept, rvalue, pvalue, std_err = stats.linregress(t, np.log(inside_log.astype('float64')))
                    # Plot fit
                    x=np.linspace(0,45)
                    y=slope*x+intercept
                    ax1.plot(x,y,color='k',linewidth=1,linestyle='--')
                    ax1.text(44,0,f'y=k*x+b\nk={slope:.5f}\nb={intercept:.5f}\nR$^2$={rvalue**2:.5f}',va='top',ha='right')
                    # Extract Fitting parameters
                    k.append(abs(slope))
                    b.append(intercept)
                
                ax2.set_title(f'{sam_type}_{element}_{r}')
                plt.savefig(f'log_plot/{sam_type}_{element}_{r}.png', dpi=1000)
    temp_dict = {'sample':sam, 'E_eq':E_eq_,'k':k, 'b':b, 'exp(b)':np.exp(b)}
    temp = pd.DataFrame(data=temp_dict, columns=temp_dict.keys())
    temp.to_csv('linfit_para.csv')

def rec_plot(df):
    """
    To check if the reaction is 21st order (~1/A) or not
    inverse of the remaining conc should be linear
    Check this with visualization
    """
    r_cand = ['r=3','r=4.5','r=6']
    for sam_type in ['Pure']:
        for element in ['Er','Nd']:
            for r in r_cand:
                fig=plt.figure()
                fig_width=3.25
                fig_height=3.25/1.618
                fig.set_size_inches(fig_width,fig_height)
                t, c = call_conc(df,sam_type, element, r)
                
                ax=fig.add_axes([0.2,0.2,0.7,0.7], xlim = (-1.5,45))
                ax.plot(t,c[0]/c,linewidth=0,marker='o')
                ax.set_title(f'1/Ln {sam_type}_{element}_{r}')
                plt.savefig(f'reciprocal_plot/{sam_type}_{element}_{r}.png', dpi=1000)


# =============================================================================
# 
#   PUBLICATION FIGURES
# 
# =============================================================================

def plot_pure_log_included(df):
    fig = plt.figure()
    fig_width = 7
    fig_height = fig_width / (1.618)
    fig.set_size_inches(fig_width, fig_height)
    
    # Axis properties
    ax1 = fig.add_axes([0.1,0.55,0.35,0.35],xlim=(-1.5,45), ylim=(-5,105))
    ax1.tick_params(direction='in', length=3)
    ax1.set_ylabel('Extraction Percentage (%)')
    #ax1.set_xlabel('Mixing Time (min)')
    ax1.text(-0.5,102.5,'(a)', weight='bold', va='top', ha='left')
    ax1.text(3.5,102.5,'Er only', weight='bold', va='top', ha='left',fontsize=10)
    
    ax2 = fig.add_axes([0.1,0.15,0.35,0.35],xlim=(-1.5,45), ylim=(-5,105))
    ax2.tick_params(direction='in', length=3)
    ax2.set_ylabel('Extraction Percentage (%)')
    ax2.set_xlabel('Mixing Time (min)')
    ax2.text(-0.5,102.5,'(b)', weight='bold', va='top', ha='left')
    ax2.text(3.5,102.5,'Nd only', weight='bold', va='top', ha='left',fontsize=10)
    
    ax3 = fig.add_axes([0.5,0.7,0.4,0.2],xlim=(0,1),ylim=(0,1))
    ax3.text(-0.01,0.95,'(c)',weight='bold',va='top', ha='right')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    ax4 = fig.add_axes([0.55,0.1,0.35,0.55])
    ax4.text(-10,0.3,'(d)', weight='bold',va='top',ha='right')
    ax4.set_xlabel('Mixing Time (min)')
    ax4.set_ylabel('Log(1-E/E$_{eq}$)')
    
    # ax1: Er extraction
    # ax2: Nd extraction
    # ax3: Cartoon expression interfacial structure
    # ax4: log
    
    def plot_pure_extraction(ax1,ax2):
        c = ['tab:blue','tab:orange','tab:green','tab:brown','tab:red']
        col_idx = 0
        df_linfit = pd.read_csv('linfit_para.csv')
        sam_type='Pure'
        r_cand = ['r=0','r=1','r=3','r=4.5','r=6'] #'r=0','r=1',
        for r in r_cand:
            # Plot Er
            t, data = call_data(df, sam_type, 'Er', r)
            ax1.errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color=c[col_idx],
                     linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
            if r=='r=4.5' and r=='r=6':
                x,y,popt,pcov = fit_exp_modified(t, data)
            else:
                x,y,popt,pcov = fit_exp(t,data)
            ax1.plot(x,y,linewidth=1, color=c[col_idx])
            
            #Plot Nd
            t, data = call_data(df, sam_type, 'Nd', r)
            ax2.errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color=c[col_idx],
                     linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
            x,y,popt,pcov = fit_exp(t,data)
            ax2.plot(x,y,linewidth=1, color=c[col_idx])
            
            col_idx+=1
        
        legends = []
        for r in r_cand:
            legends.append(f'{r}')
        ax2.legend(legends,loc='lower center', bbox_to_anchor=(0.5, -0.45),ncol=5,handletextpad=0.1,handlelength=1)
    
    def plot_pure_extraction_with_log(ax1,ax2,ax4):
        c = ['tab:blue','tab:orange','tab:green','tab:brown','tab:red']
        m = ['^','o','s']
        col_idx = 0
        mar_idx = 0
        sam_type='Pure'
        r_cand = ['r=0','r=1','r=3','r=4.5','r=6'] #'r=0','r=1',
        fit_result = {}
        for r in r_cand:
            # Plot Er
            t, E = call_data(df, sam_type, 'Er', r)
            ax1.errorbar(t,E,yerr=2**0.5*(1-0.01*E),capsize=1,elinewidth=0.75, color=c[col_idx],
                     linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
            
            if r == 'r=3':
                x,y,popt,pcov = fit_exp(t,E)
                E_eq, k, b = popt[0], popt[1], 0
                
                E=E[:-1]
                t=t[:-1]
                
                ax1.plot(x,y,linewidth=1, color=c[col_idx])
                ax4.plot(t,np.log(1-E/E_eq), linewidth=0, marker=m[mar_idx], color='red',
                         markerfacecolor='None', markeredgewidth=1, label=f'Er {r}')
                x=np.linspace(0,43)
                ax4.plot(x, -k*x+b, linewidth=1, color='k', linestyle='--',zorder=0)
            
            elif r=='r=4.5' or r=='r=6':
                x,y,popt,pcov = fit_exp_modified(t,E)
                E_eq, k, b = popt[0], popt[1], popt[2]
                
                ax1.plot(x,y,linewidth=1, color=c[col_idx])
                ax4.plot(t,np.log(1-E/E_eq), linewidth=0, marker=m[mar_idx], color='red',
                         markerfacecolor='None', markeredgewidth=1, label=f'Er {r}')
                ax4.plot(x, -k*x+b, linewidth=1, color='k', linestyle='--',zorder=0)
            
            else: # r=0 and 1
                x,y,popt,pcov = fit_exp(t,E)
                ax1.plot(x,y,linewidth=1, color=c[col_idx])
            
            #Plot Nd
            t, E = call_data(df, sam_type, 'Nd', r)
            ax2.errorbar(t,E,yerr=2**0.5*(1-0.01*E),capsize=1,elinewidth=0.75, color=c[col_idx],
                     linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
            x,y,popt,pcov = fit_exp(t,E)
            E_eq, k = popt[0], popt[1]
            ax2.plot(x,y,linewidth=1, color=c[col_idx])
            
            # Plot log (1-E/E_eq) of Nd
            if r not in ['r=0', 'r=1']:
                ax4.plot(t,np.log(1-E/E_eq), linewidth=0, marker=m[mar_idx], color='green',
                         markerfacecolor='None', markeredgewidth=1, label=f'Nd {r}')
                x=np.linspace(0,43)
                ax4.plot(x, -k*x, linewidth=1, color='k', linestyle='--',zorder=0)
                mar_idx+=1
            
            col_idx+=1
            
        
        legends = []
        for r in r_cand:
            legends.append(f'{r}')
        ax4.text(44,-0.2,'Fitted with y=-kt+b',ha='right',va='bottom',weight='bold')
        ax2.legend(legends,loc='lower center', bbox_to_anchor=(0.5, -0.45),ncol=5,handletextpad=0.1,handlelength=1)
        ax4.legend()
        
    def plot_log(ax):
        df_linfit = pd.read_csv('linfit_para.csv')
        r_cand = ['r=3', 'r=4.5', 'r=6']
        for sam_type in ['Pure']:#df['sample_type'].unique():
            for element in ['Er','Nd']:
                for r in r_cand:
                    t, c = call_conc(df,sam_type, element, r)
                    if element == 'Er':
                        color='r'
                    elif element == 'Nd':
                        color='g'
                        
                    if r == 'r=3':
                        marker='^'
                    elif r == 'r=4.5':
                        marker='o'
                    elif r== 'r=6':
                        marker='s'
                    # Plot log (1-E/Eeq)
                    t, E = call_data(df, sam_type, element, r)
                    E_eq = df_linfit[df_linfit['sample']==f'{sam_type} {element} {r}']['E_eq'].values
                    if sam_type == 'Pure' and element == 'Er' and r == 'r=3':
                        # Take the last point out: outlier
                        t = t[:-1]
                        E = E[:-1]
                        # Plot log (1-E/E_eq)
                        ax.plot(t,np.log(1-E/E_eq),linewidth=0,
                                marker=marker,color=color,markersize=3,
                                label=f'{element} {r}')
                        slope, intercept, rvalue, pvalue, std_err = stats.linregress(t, np.log(1-E/E_eq))
                        # Plot fit
                        x=np.linspace(-1,45)
                        y=slope*x+intercept
                        ax.plot(x,y,color='k',linewidth=1,linestyle='--',zorder=0)
                    else:
                        # Plot log (1-E/E_eq)
                        ax.plot(t,np.log(1-E/E_eq),linewidth=0,
                                marker=marker,color=color,markersize=3,
                                label=f'{element} {r}')
                        slope, intercept, rvalue, pvalue, std_err = stats.linregress(t, np.log(1-E/E_eq))
                        # Plot fit
                        x=np.linspace(-1,45)
                        y=slope*x+intercept
                        ax.plot(x,y,color='k',linewidth=1,linestyle='--',zorder=0)

        ax.text(44,-0.2,'Fitted with y=-kt+b',ha='right',va='bottom',weight='bold')
        ax.legend()        
    
    plot_pure_extraction_with_log(ax1,ax2,ax4)
    plt.savefig('extraction_plot/pure_wide.png', dpi=1000)


def plot_mixture(df):
    ### Plot Axis ###
    fig = plt.figure()
    fig_width = 3.25
    aspect_ratio = 3.3
    fig_height = aspect_ratio * fig_width / (1.618)
    fig.set_size_inches(fig_width, fig_height)
    
    ### Bottom
    ax1 = fig.add_axes([0.2,0.2/aspect_ratio,0.7,0.7/aspect_ratio], xlim=(-1.5,45), ylim=(-5,105))
    ax1.tick_params(direction='in', length=3)
    ax1.set_xlabel('Mixing Time (min)')
    
    ### Middle
    ax2 = fig.add_axes([0.2,1.0/aspect_ratio,0.7,0.7/aspect_ratio], xlim=(-1.5,45), ylim=(-5,105))
    ax2.tick_params(direction='in', length=3)
    ax2.set_ylabel('Extracted Percentage (%)')
    ### Top
    ax3 = fig.add_axes([0.2,1.8/aspect_ratio,0.7,0.7/aspect_ratio], xlim=(-1.5,45), ylim=(-5,105))
    ax3.tick_params(direction='in', length=3)
    ax3.text(0,105,'r=DHDP/Ln$^{3+}$', fontsize=10, weight='bold', va='bottom', ha='left')
    
    ### Box for cartoon
    ax4 = fig.add_axes([0.1,2.75/aspect_ratio,0.8,0.45/aspect_ratio])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.text(0.01,1.05, '(a)', weight='bold', va='bottom', ha='left', fontsize=10)
    
    # Plot only r=1, 3, and 6
    sam_type='Mixture'
    r_cand = ['r=1','r=3','r=6']
    label_ = ['(b)','(c)', '(d)']
    axes = [ax3, ax2, ax1]
    
    for idx, r in enumerate(r_cand):
        # Label
        axes[idx].text(0,100, label_[idx]+' '+r, weight='bold', va='top', ha='left', fontsize=10)
        # Plot Er
        element='Er'
        t,data = call_data(df, sam_type, element, r)
        axes[idx].errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color='r',
                           linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1,label='Er')
        x,y,popt,pcov = fit_exp(t,data)
        axes[idx].plot(x,y,linewidth=1, color='r')
        
        #Plot Nd
        element='Nd'
        t, data = call_data(df, sam_type, element, r)
        axes[idx].errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color='g',
                           linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1, label='Nd')
        if r == 'r=1':
            pass
        else:
            x,y,popt,pcov = fit_exp(t,data)
            axes[idx].plot(x,y,linewidth=1, color='g')
        axes[idx].legend()
    #ax1.legend()
    #ax3.set_title('Mixture Extraction\nwith DHDP/Chloroform', weight='bold')
    
    plt.savefig('extraction_plot/Mixture_cases.png', dpi=600)
    
def plot_si(df):
    # Generate Figure instance
    fig = plt.figure()
    fig_width = 7
    fig_height = fig_width / (1.618)
    fig.set_size_inches(fig_width, fig_height)

    # Set Axis
    # 2 rows (Er, Nd) and 3 columns (r=3, 4.5, 6)
    # Axis for labelling
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.set_xlabel('Mixing Time (min)')
    #ax.set_ylabel('Extraction Percentage (%)')
    ax.set_xticks([])
    ax.set_yticks([])
    for i in ['left','right','top','bottom']:
        ax.spines[i].set_visible(False)
    
    # Axis Placement
    # ax1, ax2, ax3
    # ax4, ax5, ax6
    ax_width = 0.85/3
    ax_height = 0.4
    xlim = (-1.5,45)
    ylim = (-5,105)
    ax1 = fig.add_axes([0.1,0.55,ax_width-0.01,ax_height-0.01618],xlim=xlim, ylim=ylim)
    ax2 = fig.add_axes([0.1+ax_width, 0.55, ax_width-0.01, ax_height-0.01618],xlim=xlim, ylim=ylim)
    ax3 = fig.add_axes([0.1+2*ax_width, 0.55, ax_width-0.01, ax_height-0.01618],xlim=xlim, ylim=ylim)
    ax4 = fig.add_axes([0.1,0.15,ax_width-0.01,ax_height-0.01618],xlim=xlim, ylim=ylim)
    ax5 = fig.add_axes([0.1+ax_width,0.15,ax_width-0.01,ax_height-0.01618],xlim=xlim, ylim=ylim)
    ax6 = fig.add_axes([0.1+2*ax_width,0.15,ax_width-0.01,ax_height-0.01618],xlim=xlim, ylim=ylim)
    
    axes = [ax1,ax2,ax3,ax4,ax5,ax6]
    axes_no_xticks = [ax1,ax2,ax3]
    axes_no_yticks = [ax2,ax3,ax5,ax6]
    Er_axes = [ax1,ax2,ax3]
    Nd_axes = [ax4,ax5,ax6]
     
    
    ax1.set_ylabel('Er Extraction (%)')
    ax4.set_ylabel('Nd Extraction (%)')
    for axis in axes_no_xticks:
        axis.set_xticks([])
    for axis in axes_no_yticks:
        axis.set_yticks([])
    for axis in axes:
        axis.tick_params(direction='in',length=3)
        
    def plot_Er(df, Er_axes):
        # Get data
        r_cand = ['r=3','r=4.5','r=6']
        element, sam_type = 'Er', 'Pure'
        for i, r in enumerate(r_cand):
            t,E = call_data(df, sam_type, element, r)
            Er_axes[i].set_title(f'{r}', weight='bold')
            # Plot experiment data
            Er_axes[i].plot(t,E, linewidth=0, marker='o', markerfacecolor='None',
                            markeredgewidth=1, color='red', label='Er extraction')
            # Plot without b fit
            x,y,popt,pcov = fit_exp(t, E)
            Er_axes[i].plot(x,y, linestyle='--', color='black', label='Fit without b')
            # Plot with b fit
            x,y,popt,pcov = fit_exp_modified(t,E)
            Er_axes[i].plot(x,y, color='black', label='Fit with b')
        Er_axes[-1].legend()
        
    def plot_Nd(df, Nd_axes):
        # Get data
        r_cand = ['r=3','r=4.5','r=6']
        element, sam_type = 'Nd', 'Pure'
        for i, r in enumerate(r_cand):
            t,E = call_data(df, sam_type, element, r)
            # Plot experiment data
            Nd_axes[i].plot(t,E, linewidth=0, marker='o', markerfacecolor='None',
                            markeredgewidth=1, color='green', label='Nd extraction')
            # Plot without b fit
            x,y,popt,pcov = fit_exp(t, E)
            Nd_axes[i].plot(x,y, linestyle='--', color='black', label='Fit without b')
            # Plot with b fit
            x,y,popt,pcov = fit_exp_modified(t,E)
            Nd_axes[i].plot(x,y, color='black', label='Fit with b')
        Nd_axes[-1].legend(loc='upper left')
                
    plot_Er(df, Er_axes)
    plot_Nd(df, Nd_axes)
    plt.savefig("SI_model_selection.png", dpi=1000)

# =============================================================================
# 
# FUNCTIONS TO EXECUTE
# 
# =============================================================================

def Analysis1():
    """
    Visualize with b vs without b models and their fittings
    """
    df=ms_to_conc('extraction_results')
    plot_si(df)

def Figure1():
    """
    Plot mixture Extraction results
    """
    df=ms_to_conc('extraction_results_mixture')
    plot_mixture(df)

def Figure2():
    """
    Plot Pure Extraction results
    """
    df=ms_to_conc('extraction_results')
    plot_pure_log_included(df)

# =============================================================================
# 
# MAIN
# 
# =============================================================================

if __name__ == "__main__":
    Figure1()
    Figure2()
    Analysis1()
