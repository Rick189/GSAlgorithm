#рекомендуется запускать код на Spyder, поскольку алгоритм писался на нём и при попытках запуска в других средах иногда возникали ошибки
import numpy as np
import matplotlib.pyplot as plt
import tifffile
#укажите нужную фотографию в tif формате
image=tifffile.imread('Black&White/04_Black&White.tif')
#Некоторые фотографии меньше, либо содержат больше информации, поэтому приводим их к нужному виду
if len(image.shape)>2:image=image[:,:,0]
#Выводим первое изображение для визуального сравнения с финальным результатом
plt.figure(figsize=(9,9))
plt.subplot(131)
plt.imshow(image,'gray')
plt.title("(a)")

#вычисление НСКО
def error(img):
    global image
    #деление на два у img в следствии того, что у него амплитуда в 2 раза больше
    return np.sqrt(np.sum(np.abs((np.abs(img))/2-image/1)**2)/np.sum(((image/1)**2)))
    #return np.linalg.norm(np.abs(img)-image,ord=2)/np.linalg.norm(image,ord=2)


#загружаем случайную основу и целевую амплитуду    
source=np.ones_like(image)*np.exp(1j*np.pi*2*(np.random.rand(image.shape[0],image.shape[1]))-0.5)
target=np.abs(image)

#начинаем алгоритм Гречберга-Секстона
i=0

img=source
img=np.fft.fft2(img)

#нужно для работы цикла, чтобы он не отключался сразу
err=np.inf
#массивы для графика
x=[]
y=[]

while(np.abs(err-error(img))>=0.00001):
    err=error(img)
    if(i%10==0):
        print('Iteration: '+str(i)+' Error: '+str(err))
    #считаем график
    x.append(i)
    y.append(err)
    #plt.figure()
    #plt.imshow(np.abs(img),'gray')
    #plt.title(i)
    #устанавливаем целевую амплитуду
    img=target*np.exp(1j*np.angle(img))
    #обратное преобразование
    img=np.fft.ifft2(img)
    #извлекаем фазу и устанавливаем единичную амплитуду
    img=np.exp(1j*np.angle(img))
    #переход к изображению
    img=np.fft.fft2(img)
    i+=1
print(i,error(img))
#plt.figure()
plt.subplot(132)
plt.imshow(np.abs(img)/2,'gray')
plt.title("(b)")
plt.figure(133)
plt.plot(x,y,mec='b')
plt.xlabel("i, число итераций")
plt.ylabel("E, ошибка формирования изображения")
plt.title("(c)")
