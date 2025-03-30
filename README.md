## HW1 - Singular Value Decomposition (SVD)
## 圖片壓縮應用  

### 中興大學 資料科學與資訊計算 數據分析數學
### 老師 : 施因澤 教授
### 學生 : 7113095008 江宜娉  

<br>

### source code (Python) : <https://github.com/CYP005896/NCHU-7113095008/tree/main>

<br>

### <font color="brown">**作業題目:**</font>

<br>
Assignment 1:  
Use SVD technology to select the appropriate k value to compress the photo.

1. Explain why the selected k is the optimal value? how it effects the amount of compression and the quality of the reconstructed image?

2. Use any photo matrix to prove that the image loss value = the (k+1)th singular value i.e.,  
   ‖A-A_k‖_2 = \sigma_{k+1}.

作業一：  
使用SVD技術選取適當k值將照片壓縮。

1. 說明選取的k為什麼是最佳的數值？壓縮量以及重建影像的品質？

2. 利用以上照片矩陣證明影像損失值=第(k+1)的奇異值(singular value)。

<div STYLE="page-break-after: always;"></div>

### <font color="brown">**前置準備 (圖片說明)**</font>
<br>

![alt text](image.png)
<br><br>
* 由彩圖1拆分為R、G、B三通道個別表示

![alt text](image-1.png)

```python
# 讀出一張RGB圖片(橫)
img_neveu = Image.open("neveu.jpg", "r")

# 將RGB圖片轉為矩陣表示
neveu = np.array(img_neveu)

# 將RGB圖片轉為灰階圖片
img_to_gray = Image.open("neveu.jpg").convert("L")
neveu_gray = np.array(img_to_gray)
img_gray = Image.fromarray(neveu_gray).save("neveu_gr.jpg")

# 讀出一張RGB圖片(直)
img_neveu_ver = Image.open("neveu_ver.jpg", "r")
neveu_ver = np.array(img_neveu_ver)
```

<div STYLE="page-break-after: always;"></div>

### <font color="brown">**Important functions**</font>

<br>
用2-norm、mse、psnr來做圖像品質評估統計
<br>

#### 2-norm: distance between two matrices

```python
# 2-norm
def norm2 (a, ak):
    return np.linalg.norm(a-ak, 2)
```

<br>

#### PSNR: peak signal-to-noise ratio

```python
# PSNR - 峰值訊噪比
def psnr(originalImg, sampleImg):
    # 確保輸入為浮點數以避免整數除法問題
    originalImg = originalImg.astype(np.float64)
    sampleImg = sampleImg.astype(np.float64)
    
    mse = np.mean((originalImg - sampleImg) ** 2)
    
    if mse < 1.0e-10:
        return float('inf')  # 當MSE接近0時返回無限大
    
    # 對於8位元圖片,最大像素值為255
    PIXEL_MAX = 255.0
    
    return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)
```

<br>

#### mse: mean square error

```python
# mse 均方誤差
def mse (a, ak):
    m = np.mean((a/1.0 - ak/1.0) ** 2)
    return m
```

<div STYLE="page-break-after: always;"></div>

#### Singular Value Decomposition (SVD)

```python
# Singular Value Decomposition (SVD)
def svd_restore(image, k, rgb):

    # 對圖形矩陣做 SVD 分解 : u > mxm, sigma > m, v > mxn
    u, sigma, v = np.linalg.svd(image, full_matrices=False)

    # 避免 K 值超出 sigma > m 長度
    k = min(len(sigma)-1, k) 

    # 依照 k 值，得到新的圖形矩陣
    Ak  = np.dot(u[:,:k], np.dot(np.diag(sigma[:k]), v[:k,:]))

    # 計算原圖矩陣與新圖矩陣的 2-norm
    norm = norm2(image, Ak)

    # 計算原圖矩陣與新圖矩陣的 mse
    m = mse(image, Ak)

    # value 小於 0 > 改 0
    Ak[Ak < 0] = 0     
    
    # value 大於 255 > 改 255
    Ak[Ak > 255] = 255

    # 算出來為float，四捨五入取整數，轉型為圖片所需的 uint8
    Ak = np.rint(Ak).astype('uint8')

    return Ak, sigma, norm, sigma[k], m
```

<br>

### <font color="brown">**定義K值範圍**</font>

使用的圖片大小為2147x3000、3000x4000，因此K值範圍設定為0<K<3000 (1~2999)

```python
# K值範圍
rangeArrK = np.array([1, 5, 10, 20, 50, 100, 200, 350, 500, 800, 1000, 1500, 2000, 2500, 3000])
```

<br>

### <font color="brown">**依照不同 K 值對灰階圖⽚進⾏壓縮**</font>

![alt text](image-3.png)

![alt text](image-4.png)

### <font color="brown">**作業結果：**</font>

### MSE與K值負相關，SIZE、PSNR與K值正相關

<br>

![alt text](image-6.png)

<br>

### 2-norm等於sigma(k+1)。k = 350，PSNR達到29.666039，人眼幾乎無法分辨，其為最適當的K值。

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K值</th>
      <th>2-norm</th>
      <th>sigma k+1</th>
      <th>均方誤差(MSE)</th>
      <th>峰值訊噪比(PSNR)</th>
      <th>原圖大小(Kb)</th>
      <th>圖片大小(Kb)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>52293.650118</td>
      <td>52293.650118</td>
      <td>2122.561894</td>
      <td>14.888317</td>
      <td>1067.93</td>
      <td>207.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>25570.324893</td>
      <td>25570.324893</td>
      <td>1145.742773</td>
      <td>17.591332</td>
      <td>1067.93</td>
      <td>318.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>14195.635103</td>
      <td>14195.635103</td>
      <td>867.179825</td>
      <td>18.816135</td>
      <td>1067.93</td>
      <td>391.06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>8153.680864</td>
      <td>8153.680864</td>
      <td>695.836288</td>
      <td>19.785005</td>
      <td>1067.93</td>
      <td>481.47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.0</td>
      <td>5490.910484</td>
      <td>5490.910484</td>
      <td>495.906764</td>
      <td>21.282705</td>
      <td>1067.93</td>
      <td>666.55</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.0</td>
      <td>4064.066448</td>
      <td>4064.066448</td>
      <td>324.199932</td>
      <td>23.172748</td>
      <td>1067.93</td>
      <td>812.39</td>
    </tr>
    <tr>
      <th>6</th>
      <td>200.0</td>
      <td>2528.085542</td>
      <td>2528.085542</td>
      <td>164.603834</td>
      <td>26.201542</td>
      <td>1067.93</td>
      <td>947.12</td>
    </tr>
    <tr>
      <th><span style="background-color: #FFFF00">7</span></th>
      <td><span style="background-color: #FFFF00">350.0</span></td>
      <td><span style="background-color: #FFFF00">1484.197022</span></td>
      <td><span style="background-color: #FFFF00">1484.197022</span></td>
      <td><span style="background-color: #FFFF00">75.973511</span></td>
      <td><span style="background-color: #FFFF00">29.666039</span></td>
      <td><span style="background-color: #FFFF00">1067.93</span></td>
      <td><span style="background-color: #FFFF00">1040.59</span></td>
    </tr>
    <tr>
      <th>8</th>
      <td>500.0</td>
      <td>996.857554</td>
      <td>996.857554</td>
      <td>41.628125</td>
      <td>32.305663</td>
      <td>1067.93</td>
      <td>1086.67</td>
    </tr>
    <tr>
      <th>9</th>
      <td>800.0</td>
      <td>552.929735</td>
      <td>552.929735</td>
      <td>15.790057</td>
      <td>36.060707</td>
      <td>1067.93</td>
      <td>1118.44</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1000.0</td>
      <td>408.409047</td>
      <td>408.409047</td>
      <td>8.799335</td>
      <td>38.023396</td>
      <td>1067.93</td>
      <td>1120.60</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1500.0</td>
      <td>208.124147</td>
      <td>208.124147</td>
      <td>1.742856</td>
      <td>44.366791</td>
      <td>1067.93</td>
      <td>1093.32</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000.0</td>
      <td>80.828215</td>
      <td>80.828215</td>
      <td>0.102530</td>
      <td>53.705876</td>
      <td>1067.93</td>
      <td>1070.60</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500.0</td>
      <td>36.090744</td>
      <td>36.090744</td>
      <td>0.000200</td>
      <td>inf</td>
      <td>1067.93</td>
      <td>1067.93</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3000.0</td>
      <td>36.090744</td>
      <td>36.090744</td>
      <td>0.000200</td>
      <td>inf</td>
      <td>1067.93</td>
      <td>1067.93</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### <font color="brown">**依照不同 K 值對彩色圖⽚進⾏壓縮**</font>

<br>

### method 1. Reshape method (先將三維變⼆維，⼆維做完svd，再變回三維)

```python
original_shape = neveu.shape
image_reshaped = neveu.reshape((original_shape[0], original_shape[1] * original_shape[2]))
image_reconst, sigma, norm, sigmak1, m = svd_restore(image_reshaped, k, "灰階")
image_reconst = image_reconst.reshape(original_shape)
```

#### 使用彩色圖片

![alt text](image-14.png)

<br>

![alt text](image-12.png)

![alt text](image-13.png)

<br>

### <font color="brown">**作業結果：**</font>

### MSE與K值負相關，SIZE、PSNR與K值正相關

<br>

![alt text](image-10.png)

### 2-norm等於sigma(k+1)。k = 500，PSNR達到30.921996人眼幾乎無法分辨，其為最適當的K值。

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K值</th>
      <th>2-norm</th>
      <th>sigma k+1</th>
      <th>均方誤差(MSE)</th>
      <th>峰值訊噪比(PSNR)</th>
      <th>原圖大小(Kb)</th>
      <th>圖片大小(Kb)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>92291.347675</td>
      <td>92291.347675</td>
      <td>2208.362374</td>
      <td>14.687349</td>
      <td>3415.35</td>
      <td>238.45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>44795.746905</td>
      <td>44795.746905</td>
      <td>1185.320346</td>
      <td>17.391391</td>
      <td>3415.35</td>
      <td>350.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>25113.246786</td>
      <td>25113.246786</td>
      <td>887.281792</td>
      <td>18.647548</td>
      <td>3415.35</td>
      <td>427.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>14516.715540</td>
      <td>14516.715540</td>
      <td>707.181414</td>
      <td>19.627508</td>
      <td>3415.35</td>
      <td>518.90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.0</td>
      <td>9560.577130</td>
      <td>9560.577130</td>
      <td>503.300653</td>
      <td>21.098549</td>
      <td>3415.35</td>
      <td>711.27</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.0</td>
      <td>7017.066325</td>
      <td>7017.066325</td>
      <td>330.043482</td>
      <td>22.915737</td>
      <td>3415.35</td>
      <td>861.92</td>
    </tr>
    <tr>
      <th>6</th>
      <td>200.0</td>
      <td>4417.108682</td>
      <td>4417.108682</td>
      <td>168.571695</td>
      <td>25.755786</td>
      <td>3415.35</td>
      <td>993.85</td>
    </tr>
    <tr>
      <th>7</th>
      <td>350.0</td>
      <td>2609.655572</td>
      <td>2609.655572</td>
      <td>78.118573</td>
      <td>28.825110</td>
      <td>3415.35</td>
      <td>1083.68</td>
    </tr>
    <tr>
      <th><span style="background-color: #FFFF00">8</span></th>
      <td><span style="background-color: #FFFF00">500.0</span></td>
      <td><span style="background-color: #FFFF00">1749.209178</span></td>
      <td><span style="background-color: #FFFF00">1749.209178</span></td>
      <td><span style="background-color: #FFFF00">42.899431</span></td>
      <td><span style="background-color: #FFFF00">30.921996</span></td>
      <td><span style="background-color: #FFFF00">3415.35</span></td>
      <td><span style="background-color: #FFFF00">1129.34</span></td>
    </tr>
    <tr>
      <th>9</th>
      <td>800.0</td>
      <td>971.462684</td>
      <td>971.462684</td>
      <td>16.357296</td>
      <td>33.276433</td>
      <td>3415.35</td>
      <td>1162.39</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1000.0</td>
      <td>717.781223</td>
      <td>717.781223</td>
      <td>9.163128</td>
      <td>33.966659</td>
      <td>3415.35</td>
      <td>1165.32</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1500.0</td>
      <td>368.504657</td>
      <td>368.504657</td>
      <td>1.865591</td>
      <td>34.488463</td>
      <td>3415.35</td>
      <td>1139.16</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000.0</td>
      <td>148.872682</td>
      <td>148.872682</td>
      <td>0.119081</td>
      <td>34.555996</td>
      <td>3415.35</td>
      <td>1115.71</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500.0</td>
      <td>70.939996</td>
      <td>70.939996</td>
      <td>0.000257</td>
      <td>34.558576</td>
      <td>3415.35</td>
      <td>1112.53</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3000.0</td>
      <td>70.939996</td>
      <td>70.939996</td>
      <td>0.000257</td>
      <td>34.558576</td>
      <td>3415.35</td>
      <td>1112.53</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### method 2. Layer method (分別對二維的R、G、B圖各自做SVD，再把三張圖合起來)

```python
# R的SVD分解
R, sigma, normR, sigmak1R, mR = svd_restore(neveu_ver[:, :, 0], k, "R")

# G的SVD分解  
G, sigma, normG, sigmak1G, mG = svd_restore(neveu_ver[:, :, 1], k, "G")

# B的SVD分解     
B, sigma, normB, sigmak1B, mB = svd_restore(neveu_ver[:, :, 2], k, "B")
# 壓縮後的 RGB 組合一張圖    
I = np.dstack((R, G, B))
```

#### 使用彩色圖片

![alt text](image-15.png)

![alt text](image-16.png)

<br>

### <font color="brown">**作業結果：**</font>

### MSE與K值負相關，SIZE、PSNR與K值正相關

<br>

![alt text](image-17.png)

### 2-norm等於sigma(k+1)。k = 100，PSNR達到30.677037人眼幾乎無法分辨，其為最適當的K值。

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K值</th>
      <th>2-norm</th>
      <th>sigma k+1</th>
      <th>均方誤差(MSE)</th>
      <th>峰值訊噪比(PSNR)</th>
      <th>原圖大小(Kb)</th>
      <th>圖片大小(Kb)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>97517.645344</td>
      <td>97517.645344</td>
      <td>2246.388523</td>
      <td>14.615141</td>
      <td>5743.8</td>
      <td>293.86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>36043.905000</td>
      <td>36043.905000</td>
      <td>767.618892</td>
      <td>19.283508</td>
      <td>5743.8</td>
      <td>401.47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>20242.495136</td>
      <td>20242.495136</td>
      <td>401.474859</td>
      <td>22.102544</td>
      <td>5743.8</td>
      <td>466.74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>10632.551983</td>
      <td>10632.551983</td>
      <td>214.106509</td>
      <td>24.824118</td>
      <td>5743.8</td>
      <td>549.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.0</td>
      <td>4296.599653</td>
      <td>4296.599653</td>
      <td>96.616071</td>
      <td>28.233216</td>
      <td>5743.8</td>
      <td>671.34</td>
    </tr>
    <tr>
      <th><span style="background-color: #FFFF00">5</span></th>
      <td><span style="background-color: #FFFF00">100.0</span></td>
      <td><span style="background-color: #FFFF00">2370.361416</span></td>
      <td><span style="background-color: #FFFF00">2370.361416</span></td>
      <td><span style="background-color: #FFFF00">54.210974</span></td>
      <td><span style="background-color: #FFFF00">30.677037</span></td>
      <td><span style="background-color: #FFFF00">5743.8</span></td>
      <td><span style="background-color: #FFFF00">791.38</span></td>
    </tr>
    <tr>
      <th>6</th>
      <td>200.0</td>
      <td>1285.622523</td>
      <td>1285.622523</td>
      <td>28.852934</td>
      <td>33.234677</td>
      <td>5743.8</td>
      <td>920.78</td>
    </tr>
    <tr>
      <th>7</th>
      <td>350.0</td>
      <td>779.250407</td>
      <td>779.250407</td>
      <td>16.391245</td>
      <td>35.235580</td>
      <td>5743.8</td>
      <td>1048.99</td>
    </tr>
    <tr>
      <th>8</th>
      <td>500.0</td>
      <td>566.872360</td>
      <td>566.872360</td>
      <td>10.874526</td>
      <td>36.376452</td>
      <td>5743.8</td>
      <td>1133.60</td>
    </tr>
    <tr>
      <th>9</th>
      <td>800.0</td>
      <td>374.957496</td>
      <td>374.957496</td>
      <td>5.566069</td>
      <td>37.580662</td>
      <td>5743.8</td>
      <td>1231.44</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1000.0</td>
      <td>301.803727</td>
      <td>301.803727</td>
      <td>3.671307</td>
      <td>37.986247</td>
      <td>5743.8</td>
      <td>1267.15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1500.0</td>
      <td>185.394018</td>
      <td>185.394018</td>
      <td>1.250416</td>
      <td>38.397659</td>
      <td>5743.8</td>
      <td>1306.66</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000.0</td>
      <td>109.902322</td>
      <td>109.902322</td>
      <td>0.350005</td>
      <td>38.491046</td>
      <td>5743.8</td>
      <td>1316.31</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500.0</td>
      <td>56.706875</td>
      <td>56.706875</td>
      <td>0.061550</td>
      <td>38.513253</td>
      <td>5743.8</td>
      <td>1317.51</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3000.0</td>
      <td>15.835568</td>
      <td>15.835568</td>
      <td>0.000021</td>
      <td>38.514998</td>
      <td>5743.8</td>
      <td>1317.41</td>
    </tr>
  </tbody>
</table>
</div>

### <font color="brown">**參考資料**</font>

#### 圖片來源 : 外甥小時候的照片

#### 參考程式 : [https://github.com/Chris99252/NCUC-DataAnalysis/blob/master/HW1. SVD.ipynb](<https://github.com/Chris99252/NCUC-DataAnalysis/blob/master/HW1. SVD.ipynb>)
