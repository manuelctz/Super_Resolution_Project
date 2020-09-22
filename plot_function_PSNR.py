import numpy as np
import matplotlib.pyplot as plt

baby1 = np.loadtxt('./text_files/91_images/9_1_5/N1_64_N2_32/PSNR_91_images_val_baby_64_32.txt')
bird1 = np.loadtxt('./text_files/91_images/9_1_5/N1_64_N2_32/PSNR_91_images_val_bird_64_32.txt')
butterfly1 = np.loadtxt('./text_files/91_images/9_1_5/N1_64_N2_32/PSNR_91_images_val_butterfly_64_32.txt')
head1 = np.loadtxt('./text_files/91_images/9_1_5/N1_64_N2_32/PSNR_91_images_val_head_64_32.txt')
woman1 = np.loadtxt('./text_files/91_images/9_1_5/N1_64_N2_32/PSNR_91_images_val_woman_64_32.txt')

baby15 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_16/91_images_val_baby_9115.txt')
bird15 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_16/91_images_val_bird_9115.txt')
butterfly15 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_16/91_images_val_butterfly_9115.txt')
head15 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_16/91_images_val_head_9115.txt')
woman15 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_16/91_images_val_woman_9115.txt')

baby15_32 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_32/91_images_val_baby_9115_32.txt')
bird15_32 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_32/91_images_val_bird_9115_32.txt')
butterfly15_32 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_32/91_images_val_butterfly_9115_32.txt')
head15_32 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_32/91_images_val_head_9115_32.txt')
woman15_32 = np.loadtxt('./text_files/91_images/9_1_1_5/N22_32/91_images_val_woman_9115_32.txt')

baby115 = np.loadtxt('./text_files/91_images/91115/91_images_val_baby_91115.txt')
bird115 = np.loadtxt('./text_files/91_images/91115/91_images_val_bird_91115.txt')
butterfly115 = np.loadtxt('./text_files/91_images/91115/91_images_val_butterfly_91115.txt')
head115 = np.loadtxt('./text_files/91_images/91115/91_images_val_head_91115.txt')
woman115 = np.loadtxt('./text_files/91_images/91115/91_images_val_woman_91115.txt')


x = np.array(range(1, 201))

average1 = (baby1 + bird1 + butterfly1 + head1 + woman1)/5.
average15 = (baby15 + bird15 + butterfly15 + head15 + woman15)/5.
average15_32 = (baby15_32 + bird15_32 + butterfly15_32 + head15_32 + woman15_32)/5.
average115 = (baby115 + bird115 + butterfly115 + head115 + woman115)/5.


#plt.plot(x,data1k)
plt.plot(x,average1)
plt.plot(x,average15)
plt.plot(x,average15_32)
plt.plot(x,average115)
plt.title('Model PSNR')
plt.ylabel('Average test PSNR (dB)')
plt.xlabel('Epochs')
plt.legend(['SRCNN (9-1-5)','SRCNN (9-1-1-5, $n_{22}$=16)','SRCNN (9-1-1-5, $n_{22}$=32)','SRCNN (9-1-1-1-5, $n_{22}$=32, $n_{23}$=16)'], loc='lower right')
plt.xlim(1,200)
plt.ylim(27,34)
plt.grid()
plt.show()