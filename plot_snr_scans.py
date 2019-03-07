import numpy
import myplot
import matplotlib.pyplot as plt

if __name__=='__main__':

    p1 = '/home/ejo/aso/trigger/snr_scan_files/32window/'
    p2 = '/home/ejo/aso/trigger/snr_scan_files/16_window/'


    label1 = 'window=32'
    label2 = 'window=16'
    plt.figure(4,figsize=(8.5,7))

    plt.figure(1,figsize=(8.5,7))
    
    dat1 = numpy.loadtxt(p1+'1phisector_phi15_theta-10.txt')
    dat2 = numpy.loadtxt(p2+'1phisector_phi15_theta-10_notop.txt')

    plt.plot(dat1[0], dat1[3], label=label1)
    plt.plot(dat2[0], dat2[3], label=label2)
    
    plt.figure(4)
    plt.plot(dat1[0], dat1[3], label='1 phi sector')

    plt.figure(1)
    plt.title('Single phi-sector coherent sum', fontsize=20)
    plt.grid(True)
    plt.ylim([-.1,1.1])
    plt.xlim([0,4])
    plt.xlabel('peak single-antenna voltage SNR')
    plt.ylabel('Trigger effc')
    plt.legend(loc='upper left')
    

    plt.figure(2,figsize=(8.5,7))
    
    dat1 = numpy.loadtxt(p1+'3phisector_phi15_theta-10.txt')
    dat2 = numpy.loadtxt(p2+'3phisector_phi15_theta-10_notop.txt')

    plt.plot(dat1[0], dat1[3], label=label1)
    plt.plot(dat2[0], dat2[3], label=label2)

    plt.figure(4)
    plt.plot(dat1[0], dat1[3], label='3 phi sector')

    plt.figure(2)
    plt.title('Three phi-sector coherent sum', fontsize=20)
    plt.grid(True)
    plt.ylim([-.1,1.1])
    plt.xlim([0,4])
    plt.xlabel('peak single-antenna voltage SNR')
    plt.ylabel('Trigger effc')
    plt.legend(loc='upper left')

    plt.figure(3,figsize=(8.5,7))
    
    dat1 = numpy.loadtxt(p1+'4phisector_phi22.5_theta-10.txt')
    dat2 = numpy.loadtxt(p2+'4phisector_phi22.5_theta-10_notop.txt')

    plt.plot(dat1[0], dat1[3], label=label1)
    plt.plot(dat2[0], dat2[3], label=label2)

    plt.figure(4)
    plt.plot(dat1[0], dat1[3], label='4 phi sector')

    plt.figure(3)
    plt.title('Four phi-sector coherent sum', fontsize=20)
    plt.grid(True)
    plt.ylim([-.1,1.1])
    plt.xlim([0,4])
    plt.xlabel('peak single-antenna voltage SNR')
    plt.ylabel('Trigger effc')
    plt.legend(loc='upper left')

    plt.figure(4)
    dat1 = numpy.loadtxt(p1+'single_antenna_phi15_theta-10.txt')
    plt.plot(dat1[0], dat1[3], label='single antenna')

    plt.grid(True)
    plt.ylim([-.1,1.3])
    plt.xlim([0,6.0])
    plt.xlabel('peak single-antenna voltage SNR')
    plt.ylabel('Trigger effc')
    plt.legend(loc='upper left',ncol=2)
    
    
    plt.show()
