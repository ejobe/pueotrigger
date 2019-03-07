import matplotlib as mpl

#mpl.use('Qt4Agg') #for use on midway
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rc('legend',**{'fontsize':16})

mpl.rcParams['xtick.major.size'] = 8.5 
mpl.rcParams['ytick.major.size'] = 8.5                               
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.major.width'] = 1       
mpl.rcParams['ytick.minor.width'] = 1       
mpl.rcParams['xtick.major.width'] = .5       
mpl.rcParams['xtick.minor.width'] = .5  

##these only work in 1.5.3+
#mpl.rcParams['xtick.minor.visible'] = True
#mpl.rcParams['ytick.minor.visible'] = True

mpl.rcParams['lines.linewidth'] = 1.5

mpl.rcParams['axes.linewidth'] = 1.8
mpl.rcParams['axes.labelsize'] = 16
##only works in 1.5.3+
#mpl.rcParams['axes.labelpad'] = 6
  

