# Processamento OCR para o projeto WASS
# na praia do Leme/RJ
# Descricao: Foram filmados 2 relogios com os
# 2 celulares
# Henrique Pereira
# Matheus Vieira
#
# convert -delay 20 -loop 0 *.jpg myimage.gif


# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import cv2
import pytesseract as ocr
from PIL import Image

# If you don't have tesseract executable in your PATH, include the following:
# ocr.pytesseract.tesseract_cmd = r'/usr/local/python/anaconda3/lib/python3.6/site-packages/pytesseract/'
# ocr.pytesseract.tesseract_cmd = r'/usr/local/python/anaconda3/lib/python3.7/site-packages/pytesseract/'

plt.close('all')

def quality_control(time_ocr0, time_ocr1):
    """
    """

    # corrige valores com string nao numerica colocando um valor posterior ao valor atual
    for t in range(len(time_ocr0)):
         # se nao for numerico
         if time_ocr0.values[t][0][-1].isnumeric() == False:
             time_ocr0.values[t][0] = time_ocr0.values[t][0][:-1] + str(int(time_ocr0.values[t-1][0][-1])+1)
             # time_ocr0.values[t][0] = np.nan
         # se nao for sequencial
         if int(time_ocr0.values[t-1][0][-1]) != int(time_ocr0.values[t][0][-1]) + 1:
             time_ocr0.values[t][0] = time_ocr0.values[t][0][:-1] + str(int(time_ocr0.values[t-1][0][-1])+1)

    # for t in range(len(time_ocr1)):
    #     # se nao for numerico
    #     if time_ocr1.values[t][0][-1].isnumeric() == False:
    #         time_ocr1.values[t][0] = time_ocr1.values[t][0][:-1] + str(int(time_ocr1.values[t-1][0][-1])+1)
    #     # se nao for sequencial
    #     if int(time_ocr1.values[t-1][0][-1]) != int(time_ocr1.values[t][0][-1]) + 1:
    #         time_ocr1.values[t][0] = time_ocr1.values[t][0][:-1] + str(int(time_ocr1.values[t-1][0][-1])+1)

    # correcao de valores pontuais
    time_ocr0[0][1156] = '02:01:9'
    time_ocr0[0][1526] = '02:38:9'
    time_ocr0.iloc[np.where(time_ocr0 == '00:44:§')] = '00:44:3'
    time_ocr0.iloc[np.where(time_ocr0 == '02:0729')] = '02:07:29'
    time_ocr0.iloc[np.where(time_ocr0 == '02:3385')] = '02:33:85'
    time_ocr0.iloc[np.where(time_ocr0 == '00:45:S')] = '00:45:3'

    return time_ocr0, time_ocr1

def plot_contours(path_fig, im0, im1, phrase0, phrase1, f):
    """
    """

    # digits = np.array(digits).astype(int)
    # [cv2.putText(im, str(int(digits[i])), (0, 0),cv2.FONT_HERSHEY_DUPLEX, 4.6, (0, 255, 255), 2) for i in range(len(digits))]
    # cv2.putText(im, str(int(digits)), (0, 0),cv2.FONT_HERSHEY_DUPLEX, 4.6, (0, 255, 255), 2)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_title(phrase0, fontsize=25)
    ax1.imshow(im0)
    ax2 = fig.add_subplot(212)
    ax2.set_title(phrase1, fontsize=25)
    ax2.imshow(im1)
    fig.savefig('{}frame_ocr_{}.png'.format(path_fig, f))
    plt.close('all')
    return

if __name__ == "__main__":

    timei = 100
    timef = 10*60*4
    # timef = 20

    do_make_ocr = True
    do_read_ocr = False

    path_video = os.environ['HOME'] + '/gdrive/coppe/lioc/wass/data/ocr/'
    path_fig = os.environ['HOME'] + '/Documents/wass/figs/ocr/'
    path_out = os.environ['HOME'] + '/gdrive/coppe/lioc/wass/out/'

    file_v0 = 'cam0_fps10_sync.mp4'
    file_v1 = 'cam1_fps10_sync.mp4'

    cap0 = cv2.VideoCapture(path_video + file_v0)
    dim0 = [693,828,308,854]
    # dim0 = [420,570,487,1027]
    ncam0 = 0

    cap1 = cv2.VideoCapture(path_video + file_v1)
    dim1 = [744,888,45,600]
    ncam1 = 1

    if do_make_ocr:
        print ('making ocr..')

        plt.close('all')
        
        time_ocr0 = [] # tempo retirado da imagem em string
        list_digits0 = [] # lista com os digitos
        time_ocr1 = [] # tempo retirado da imagem em string
        list_digits1 = [] # lista com os digitos

        for f in np.arange(timei, timef, 4):
        #for f in np.arange(100, 120, 1):
            print (f)

            cap0.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret0, frame0 = cap0.read()

            cap1.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret1, frame1 = cap1.read()

            im0 = frame0[dim0[0]:dim0[1], dim0[2]:dim0[3]]
            im1 = frame1[dim1[0]:dim1[1], dim1[2]:dim1[3]]

            cv2.imwrite(path_fig + 'frame0_{}.png'.format(ncam0), im0)
            cv2.imwrite(path_fig + 'frame1_{}.png'.format(ncam1), im1)

            phrase0 = ocr.image_to_string(Image.open(path_fig + 'frame0_{}.png'.format(ncam0)),
                lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

            phrase1 = ocr.image_to_string(Image.open(path_fig + 'frame1_{}.png'.format(ncam1)),
                lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

            phrase0 = phrase0.replace(' ','')[:7]
            phrase1 = phrase1.replace(' ','')[:7]

            # digits0 = [phrase0[0],phrase0[1],phrase0[3],phrase0[4],phrase0[6]]
            # digits1 = [phrase1[0],phrase1[1],phrase1[3],phrase1[4],phrase1[6]]

            print (phrase0)
            print (phrase1)

            time_ocr0.append(phrase0[:7])
            time_ocr1.append(phrase1[:7])

            plot_contours(path_fig, im0, im1, phrase0, phrase1, f)

        cap0.release()
        cap1.release()

        df_time_ocr0 = pd.DataFrame(np.array(time_ocr0))
        df_time_ocr1 = pd.DataFrame(np.array(time_ocr1))

        #time_ocr0_qc, time_ocr1_qc = quality_control(df_time_ocr0, df_time_ocr1)

        #df_time_ocr0_dt = pd.to_datetime(time_ocr0[0], format='%M:%S:%f')[36:2338]
        #df_time_ocr1_dt = pd.to_datetime(time_ocr1[0], format='%M:%S:%f')[36:2338]

        df_time_ocr0_dt = pd.to_datetime(time_ocr0, format='%M:%S:%f')
        df_time_ocr1_dt = pd.to_datetime(time_ocr1, format='%M:%S:%f')

        # converte para segundos a diferenca
        # a = time_ocr1_dt_qc - time_ocr0_dt_qc
        # b = a.dt.total_seconds

        df_time_ocr0_dt.to_csv(path_out + 'time_ocr0_dt.csv')
        df_time_ocr1_dt.to_csv(path_out + 'time_ocr1_dt.csv')

    if do_read_ocr:
        print ('reading output..')

        time_ocr0_dt = pd.read_csv(path_out + 'time_ocr0_dt.csv', index_col=0, header=None)
        time_ocr1_dt = pd.read_csv(path_out + 'time_ocr1_dt.csv', index_col=0, header=None)

        time_ocr0_dt[1] = pd.to_datetime(time_ocr0_dt[1])
        time_ocr1_dt[1] = pd.to_datetime(time_ocr1_dt[1])

        time_dif = time_ocr1_dt[1] - time_ocr0_dt[1]

        time_dif_sec = time_dif.dt.total_seconds()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(time_ocr0_dt)
        ax1.plot(time_ocr1_dt)

        plt.show()


    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(time_ocr0,'-o', markersize=6, label='CAM_0')
    # ax1.plot(time_ocr1,'-o', markersize=3, label='CAM_1')
    # ax1.set_xlabel('Nº Frame (10 FPS)')
    # ax1.set_ylabel('Time (MM:SS)')
    # plt.grid()
    # ax1.yaxis.set_major_formatter(DateFormatter("%M:%S"))
    # ax1.legend()
